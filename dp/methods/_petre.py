from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple
import string
import numpy as np

from dp.loaders import DatasetRecord
from dp.methods.anonymizer import AnonymizationResult, Anonymizer
from dp.methods.constants import Buckets, BucketDict, buckets_to_dicts
from dp.utils.splitter import TextSplitter
from dp.utils.memory import clear_memory
from dp.utils.token_ledger import TokenLedger
from dp.utils.stopwords import build_terms_to_ignore
from dp.loaders.base import TextAnnotation, TextAnnotations, TokenEdit
from dp.utils.explainer.base import load_tri_label_mapping
from dp.utils.selector.base import AnonymizerUnit, ApplyFn
from dp.utils.precomputed_risk import align_precomputed_risk_scores


class PetreAnonymizer(Anonymizer):   
    MODEL_NAME = "petre"
    def __init__(
        self,
        *args,
        mask_text: str = "[MASK]",
        **kwargs
    ):
        super().__init__(*args, model=self.MODEL_NAME, **kwargs)

        self.mask_text = mask_text
        self.ignore_terms = build_terms_to_ignore(self.mask_text)

        self._unit: Optional[AnonymizerUnit] = None
        self._explainer = None
        self.splitter = TextSplitter()
        self._risk_scores_by_uid: Dict[str, Dict[Tuple[int, int], float]] = {}
        self._risk_offsets_by_uid: Dict[str, List[Tuple[int, int]]] = {}
        self.dataset_records: List[DatasetRecord] = []

        self._tri_label_mapping: Optional[Dict[str, int]] = None
        self._tri_label_mapping_source: Optional[str] = None

        self._pre_stream_starting_by_uid: Dict[
            str,
            Tuple[
                List[Tuple[int, int]],
                List[int],
                Dict[int, str],
                Dict[int, str],
            ],
        ] = {}

        self._pre_stream_direct_ledger_by_uid: Dict[str, TokenLedger] = {}

    def set_unit(self, unit: AnonymizerUnit) -> None:
        self._unit = unit

    def set_risk_scores(
        self,
        risk_scores: Dict[str, Dict[str, object]],
        records: Optional[Sequence[DatasetRecord]] = None,
    ) -> None:
        self._risk_scores_by_uid = {}
        self._risk_offsets_by_uid = {}
        resolved = align_precomputed_risk_scores(risk_scores, records=records)
        for uid, entry in resolved.items():
            self._risk_scores_by_uid[uid] = entry.span_scores
            self._risk_offsets_by_uid[uid] = entry.ordered_offsets

    def add_dataset_records(self, dataset_records: Sequence[DatasetRecord]) -> None:
        self.dataset_records = list(dataset_records)

    def pre_stream_anonymize(
        self,
        *args,
        risk_scores: Optional[Dict[str, Dict[str, object]]] = None,
        **kwargs,
    ) -> None:
        if risk_scores is not None:
            if not self.dataset_records:
                raise ValueError(
                    "PetreAnonymizer received precomputed risk_scores but has no dataset records; "
                    "run in dataset mode (use --indices) so risk scores can be matched to records"
                )
            self.set_risk_scores(risk_scores, records=self.dataset_records or None)

    def anonymize_from_dataset(
        self,
        idx: int,
        *args,
        buckets: Buckets = [],
        **kwargs,
    ) -> List[Tuple[BucketDict, AnonymizationResult]]:
        if idx < 0 or idx >= len(self.dataset_records):
            raise IndexError(f"Index {idx} is out of bounds")
        record = self.dataset_records[idx]
        
        masked_spans = record.spans or []
        
        return self.anonymize_any_text(
            record.text,
            *args,
            buckets=buckets,
            record_name=record.name,
            record_uid=str(record.uid),
            masked_spans=masked_spans,
            **kwargs,
        )

    def _target_label_id_for_record(self, record_name: Optional[str]) -> int:
        if not record_name:
            raise ValueError("record_name is required for TRI rank evaluation")
        mapping, source = load_tri_label_mapping(self._explainer, self._tri_label_mapping, self._tri_label_mapping_source)
        self._tri_label_mapping = mapping
        self._tri_label_mapping_source = source
        if record_name not in mapping:
            raise ValueError("record_name not present in TRI label mapping")
        return int(mapping[record_name])

    def _make_rank_evaluator(self) -> Callable[[str, int], int]:
        if self._explainer is None:
            raise ValueError("PetreAnonymizer requires explainer for rank evaluation")
        if hasattr(self._explainer, "_load_pipeline"):
            self._explainer._load_pipeline()
        predict_entries = getattr(self._explainer, "predict_entries", None)
        pipe = getattr(self._explainer, "pipeline", None)
        if predict_entries is None and pipe is None:
            raise ValueError("Explainer predictions are not available for rank evaluation")

        def rank_evaluator(current_text: str, target_label: int) -> int:
            target = f"LABEL_{int(target_label)}"
            if callable(predict_entries):
                entries = predict_entries([current_text], batch_size=1)[0]
            else:
                entries = pipe([current_text], batch_size=1)[0]
            if not isinstance(entries, list) or not entries:
                raise ValueError("TRI pipeline returned no predictions")
            scored = [e for e in entries if isinstance(e, dict) and "label" in e and "score" in e]
            if not scored:
                raise ValueError("TRI pipeline returned invalid predictions")
            scored.sort(key=lambda e: float(e["score"]), reverse=True)
            for i, e in enumerate(scored, start=1):
                if str(e.get("label")) == target:
                    return i
            raise ValueError("Target label not found in TRI predictions")

        return rank_evaluator

    def _lookup_precomputed_scores(
        self,
        text: str,
        offsets: Sequence[Tuple[int, int]],
        record_name: Optional[str],
        record_uid: Optional[str] = None,
        indices: Optional[Sequence[int]] = None,
    ) -> Optional[np.ndarray]:
        uid = self._resolve_risk_uid(text, record_name, record_uid)
        if uid is None:
            return None
        mapping = self._risk_scores_by_uid.get(uid)
        if mapping is None:
            return None
        values: List[float] = []
        for idx in range(len(offsets)):
            if idx < 0 or idx >= len(offsets):
                return None
            if indices is not None and idx not in indices:
                continue
            span = offsets[idx]
            key = (int(span[0]), int(span[1]))
            if key not in mapping:
                return None
            values.append(float(mapping[key]))
        if not values:
            return None
        return np.asarray(values, dtype=float)

    def _resolve_risk_uid(self, text: str, record_name: Optional[str], record_uid: Optional[str] = None) -> Optional[str]:
        if record_uid and record_uid in self._risk_scores_by_uid:
            return record_uid
        if record_name and record_name in self._risk_scores_by_uid:
            return record_name
        return None

    def _resolve_precomputed_offsets(
        self,
        record_name: Optional[str],
        record_uid: Optional[str],
    ) -> Optional[List[Tuple[int, int]]]:
        if record_uid and record_uid in self._risk_offsets_by_uid:
            return self._risk_offsets_by_uid[record_uid]
        if record_name and record_name in self._risk_offsets_by_uid:
            return self._risk_offsets_by_uid[record_name]
        return None

    def _make_apply_fn(
        self,
        text: str,
        offsets: List[Tuple[int, int]],
        runtime_stats: Dict[str, int],
        masked_spans: Sequence[TextAnnotation],
    ) -> ApplyFn:
        def apply_fn(idx: int, ledger: TokenLedger) -> None:
            if idx >= len(offsets):
                return
            entry = ledger.entry(idx)
            token = entry.original_text
            if self._is_token_masked(entry.start, entry.end, masked_spans):
                runtime_stats["total"] += 1
                return
            if token in self.ignore_terms or token in string.punctuation:
                runtime_stats["total"] += 1
                return
            ledger.replace(idx, self.mask_text)
            runtime_stats["masked"] += 1
            runtime_stats["total"] += 1
        return apply_fn

    def _collect_risk_scores(
        self,
        text: str,
        offsets: Sequence[Tuple[int, int]],
        record_name: Optional[str],
        record_uid: Optional[str] = None,
        critical_indices: Optional[Sequence[int]] = None,
    ) -> Tuple[np.ndarray, bool]:
        precomputed_scores = self._lookup_precomputed_scores(text, offsets, record_name, record_uid=record_uid, indices=critical_indices)
        if precomputed_scores is not None:
            return precomputed_scores, True

        critical_offsets = offsets if critical_indices is None else [offsets[i] for i in critical_indices]
        from dp.utils.explainer.uniform import UniformExplainer
        if self._explainer is not None and isinstance(self._explainer, UniformExplainer):
            scores = self._explainer.explain(text, critical_offsets)
            return scores, False

        raise NotImplementedError("PetreAnonymizer requires precomputed risk scores for each record.")

    def _is_token_masked(self, token_start: int, token_end: int, masked_spans: Sequence[TextAnnotation]) -> bool:
        for span in masked_spans:
            if token_start >= span.start and token_end <= span.end:
                return True
        return False

    def anonymize_any_text(
        self,
        text: str,
        *args,
        buckets: Buckets = [],
        record_name: Optional[str] = None,
        record_uid: Optional[str] = None,
        masked_spans: Optional[Sequence[TextAnnotation]] = None,
        **kwargs,
    ) -> List[Tuple[BucketDict, AnonymizationResult]]:
        if not text or not text.strip():
            return []
        
        if masked_spans is None:
            masked_spans = []

        combos = buckets_to_dicts(buckets)
        outputs: List[Tuple[BucketDict, AnonymizationResult]] = []

        for hp in combos:
            try:
                precomputed_offsets = self._resolve_precomputed_offsets(record_name, record_uid)
                if precomputed_offsets is None:
                    raise ValueError("PetreAnonymizer requires precomputed offsets for each record")
                else:
                    offsets = precomputed_offsets

                from dp.utils.selector.until_k_selector import UntilKUnit
                if self._unit is None or not isinstance(self._unit, UntilKUnit):
                    self._unit = UntilKUnit()

                k_val = hp.get("k")
                if k_val is None:
                    raise ValueError("PetreAnonymizer using until_k selector requires KParams buckets")
                self._unit.set_thresholds([int(k_val)], name="k")

                target_label_id = self._target_label_id_for_record(record_name)
                self._unit.set_target_label(target_label_id)
                self._unit.set_rank_evaluator(self._make_rank_evaluator())

                context: Dict[str, Any] = {"record_name": record_name}
                used_precomputed = False

                risk_scores, used_precomputed = self._collect_risk_scores(text, offsets, record_name, record_uid=record_uid)
                if risk_scores.size and len(risk_scores) == len(offsets):
                    self._unit.set_risk_scores(risk_scores)

                runtime_stats: Dict[str, int] = {
                    "total": 0,
                    "masked": 0,
                }

                ledger = TokenLedger(text, offsets)

                apply_fn = self._make_apply_fn(
                    text,
                    offsets,
                    runtime_stats,
                    masked_spans,
                )

                for step in self._unit.anonymize(text, offsets, apply_fn, ledger, **context):
                    hp_with_threshold = {**hp}
                    threshold = step.threshold

                    private_text = step.text
                    ledger = step.ledger

                    metadata: Dict[str, Any] = {
                        "method": "petre",
                        "threshold": threshold,
                        **runtime_stats,
                        **step.metadata,
                    }
                    token_edits = [TokenEdit.from_mapping(e) for e in ledger.result_edits_metadata()]
                    if used_precomputed:
                        metadata["explainer"] = "PrecomputedRisk"
                    elif self._explainer is not None:
                        metadata["explainer"] = self._explainer.__class__.__name__

                    outputs.append((
                        hp_with_threshold,
                        AnonymizationResult(
                            text=private_text,
                            annotations=TextAnnotations(
                                spans=[],
                                token_edits=token_edits,
                            ),
                            metadata=metadata,
                        ),
                    ))

            finally:
                clear_memory()

        return outputs
