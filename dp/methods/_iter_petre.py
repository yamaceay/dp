import math
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np
import string

from dp.methods._petre import PetreAnonymizer
from dp.methods.anonymizer import AnonymizationResult
from dp.methods.constants import Buckets, BucketDict, buckets_to_dicts
from dp.utils.token_ledger import TokenLedger
from dp.utils.memory import clear_memory
from dp.loaders.base import TextAnnotation, TextAnnotations, TokenEdit


class IterPetreAnonymizer(PetreAnonymizer):
    MODEL_NAME = "iter_petre"

    def __init__(self, *args, T: Union[int, float] = math.inf, verbose: bool = False, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if T != math.inf and (not isinstance(T, int) or T < 1):
            raise ValueError("T must be a positive integer or math.inf")
        self.T = T
        self.verbose = bool(verbose)

    def _compute_current_offsets(self, ledger: TokenLedger, n: int) -> List[Tuple[int, int]]:
        result: List[Optional[Tuple[int, int]]] = [None] * n
        sorted_entries = sorted(ledger._entries, key=lambda e: e.start)
        cursor = 0
        gap_idx = 0
        gaps = ledger._gaps

        for entry in sorted_entries:
            while gap_idx < len(gaps) and gaps[gap_idx].start < entry.start:
                cursor += len(gaps[gap_idx].text)
                gap_idx += 1
            if not entry.deleted:
                result[entry.index] = (cursor, cursor + len(entry.text))
                cursor += len(entry.text)

        for i in range(n):
            if result[i] is None:
                result[i] = (0, 0)

        return result  # type: ignore

    def _refresh_shap_scores(
        self,
        ledger: TokenLedger,
        original_text: str,
        n: int,
        target_label_str: str,
    ) -> np.ndarray:
        current_text = ledger.render_offsets(original_text)
        current_offsets = self._compute_current_offsets(ledger, n)

        surviving = [(i, current_offsets[i]) for i in range(n) if not ledger.entry(i).deleted]
        if not surviving:
            return np.zeros(n, dtype=float)

        surviving_indices, surviving_offsets = zip(*surviving)
        raw_scores = self._explainer.explain(
            current_text, list(surviving_offsets), target_label=target_label_str
        )

        scores = np.zeros(n, dtype=float)
        for pos, orig_idx in enumerate(surviving_indices):
            scores[orig_idx] = float(raw_scores[pos])
        return scores

    def _print_step_risk_scores(
        self,
        step: int,
        shap_scores: np.ndarray,
        unprocessed: set,
        text: str,
        offsets: List[Tuple[int, int]],
        record_ref: str,
        refreshed: bool,
    ) -> None:
        if not self.verbose:
            return
        top = sorted(unprocessed, key=lambda i: float(shap_scores[i]), reverse=True)[:5]
        refresh_marker = " [refreshed]" if refreshed else ""
        tokens = [f"{text[offsets[i][0]:offsets[i][1]]}({shap_scores[i]:.4f})" for i in top]
        print(
            f"[iter_petre][verbose] record={record_ref} step={step}{refresh_marker} "
            f"remaining={len(unprocessed)} top5={tokens}"
        )

    def _apply_mask(
        self,
        idx: int,
        ledger: TokenLedger,
        masked_spans: Sequence[TextAnnotation],
        runtime_stats: Dict[str, int],
    ) -> None:
        entry = ledger.entry(idx)
        if self._is_token_masked(entry.start, entry.end, masked_spans):
            runtime_stats["total"] += 1
            return
        token = entry.original_text
        if token in self.ignore_terms or token in string.punctuation:
            runtime_stats["total"] += 1
            return
        ledger.replace(idx, self.mask_text)
        runtime_stats["masked"] += 1
        runtime_stats["total"] += 1

    def _build_output(
        self,
        hp: BucketDict,
        ledger: TokenLedger,
        original_text: str,
        offsets: List[Tuple[int, int]],
        threshold_type: Optional[str],
        threshold: Any,
        used_precomputed: bool,
        runtime_stats: Dict[str, int],
        extra_meta: Dict[str, Any],
    ) -> Tuple[BucketDict, AnonymizationResult]:
        hp_out = {**hp}
        if threshold is not None and threshold_type in {"k"}:
            hp_out[threshold_type] = threshold

        private_text = ledger.render_offsets(original_text)
        token_edits = [TokenEdit.from_mapping(e) for e in ledger.result_edits_metadata()]

        if used_precomputed:
            explainer_name = "PrecomputedRisk"
        elif self._explainer is not None:
            explainer_name = self._explainer.__class__.__name__
        else:
            explainer_name = "unknown"

        metadata: Dict[str, Any] = {
            "method": self.MODEL_NAME,
            "threshold": threshold,
            "explainer": explainer_name,
            "T": self.T,
            **runtime_stats,
            **extra_meta,
        }
        metadata["total"] = int(len(offsets))

        return (
            hp_out,
            AnonymizationResult(
                text=private_text,
                annotations=TextAnnotations(spans=[], token_edits=token_edits),
                metadata=metadata,
            ),
        )

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

        precomputed_offsets = self._resolve_precomputed_offsets(record_name, record_uid)
        if precomputed_offsets is None:
            raise ValueError("IterPetreAnonymizer requires precomputed offsets for each record")
        offsets = precomputed_offsets
        n = len(offsets)

        static_shap_scores: Optional[np.ndarray] = None
        used_precomputed = False
        if math.isinf(self.T):
            static_shap_scores = self._lookup_precomputed_scores(
                text, offsets, record_name, record_uid=record_uid
            )
            used_precomputed = static_shap_scores is not None

        needs_live_shap = static_shap_scores is None
        if needs_live_shap and self._explainer is None:
            raise ValueError(
                "IterPetreAnonymizer requires precomputed risk scores (for T=inf) or an explainer"
            )

        target_label_id = self._target_label_id_for_record(record_name)
        target_label_str = f"LABEL_{target_label_id}"
        rank_evaluator = self._make_rank_evaluator()

        for hp in combos:
            try:
                k_val = hp.get("k")
                if k_val is None:
                    raise ValueError("IterPetreAnonymizer requires KParams buckets")
                target_k = int(k_val)

                runtime_stats: Dict[str, int] = {"total": 0, "masked": 0}
                ledger = TokenLedger(text, offsets)
                shap_scores: Optional[np.ndarray] = static_shap_scores
                step_count = 0
                unprocessed: set = set(range(n))

                current_text = ledger.render_offsets(text)
                current_rank = rank_evaluator(current_text, target_label_id)

                if current_rank >= target_k:
                    outputs.append(self._build_output(
                        hp=hp, ledger=ledger, original_text=text, offsets=offsets,
                        threshold_type="k", threshold=target_k,
                        used_precomputed=used_precomputed, runtime_stats=runtime_stats,
                        extra_meta={"rank": current_rank, "processed_count": 0},
                    ))
                    continue

                while unprocessed and current_rank < target_k:
                    needs_refresh = shap_scores is None or (
                        not math.isinf(self.T) and step_count % int(self.T) == 0
                    )
                    if needs_refresh:
                        shap_scores = self._refresh_shap_scores(
                            ledger, text, n, target_label_str
                        )

                    record_ref = record_uid or record_name or "<unknown>"
                    self._print_step_risk_scores(
                        step_count, shap_scores, unprocessed, text, offsets,
                        record_ref, needs_refresh,
                    )

                    candidates = sorted(
                        unprocessed,
                        key=lambda i: float(shap_scores[i]),
                        reverse=True,
                    )
                    idx = candidates[0]

                    self._apply_mask(idx, ledger, masked_spans, runtime_stats)
                    unprocessed.discard(idx)
                    step_count += 1

                    if ledger.is_modified(idx):
                        current_text = ledger.render_offsets(text)
                        current_rank = rank_evaluator(current_text, target_label_id)

                outputs.append(self._build_output(
                    hp=hp, ledger=ledger, original_text=text, offsets=offsets,
                    threshold_type="k", threshold=target_k,
                    used_precomputed=used_precomputed, runtime_stats=runtime_stats,
                    extra_meta={"rank": current_rank, "processed_count": n - len(unprocessed)},
                ))

            finally:
                clear_memory()

        return outputs
