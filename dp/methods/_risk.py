from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
from hashlib import sha256

import numpy as np

from dp.loaders import TextAnnotation
from dp.loaders.base import TextAnnotations, TokenEdit
from dp.methods.anonymizer import AnonymizationResult, Anonymizer
from dp.methods.constants import Buckets, RhoParams
from dp.utils.splitter import TextSplitter
from dp.utils.token_ledger import TokenLedger
from dp.utils.selector.base import AnonymizerUnit, ApplyFn
from dp.utils.selector.by_risk_selector import ByRiskUnit
from dp.utils.stopwords import build_terms_to_ignore
from dp.utils.precomputed_risk import align_precomputed_risk_scores
from dp.loaders import DatasetRecord


class RiskAnonymizer(Anonymizer):
    MODEL_NAME = "risk"
    
    def __init__(self, *args, risk_temperature: float = 1.0, mask_text: str = "[MASK]", **kwargs):
        super().__init__(*args, model=self.MODEL_NAME, **kwargs)
        if risk_temperature <= 0:
            raise ValueError("temperature must be positive")
        self._mask_text = mask_text
        self._temperature = float(risk_temperature)
        self._splitter = TextSplitter()
        self._unit: Optional[ByRiskUnit] = ByRiskUnit(temperature=self._temperature)
        self._risk_scores_by_uid: Dict[str, Dict[Tuple[int, int], float]] = {}
        self._risk_offsets_by_uid: Dict[str, List[Tuple[int, int]]] = {}
        self._risk_scores_ordered_by_uid: Dict[str, np.ndarray] = {}
        self._scores_cache: Dict[str, Tuple[np.ndarray, List[Tuple[int, int]]]] = {}
        self._starting_indices_cache: Dict[str, List[int]] = {}
        self._prior_edits_cache: Dict[str, List[Dict[str, object]]] = {}
        self._terms_to_ignore = build_terms_to_ignore(self._mask_text)
        self.dataset_records: List[DatasetRecord] = []

    def set_unit(self, unit: AnonymizerUnit) -> None:
        self._unit = unit
    
    def hash_text(self, text: str) -> str:
        return sha256(text.encode('utf-8')).hexdigest()
    
    def pre_stream_anonymize(self, *args, texts_or_indices: Optional[Union[List[str], List[int]]] = None, risk_scores: Optional[Dict[str, Dict[str, object]]] = None, **kwargs) -> None:
        if not all(isinstance(i, str) for i in texts_or_indices):
            raise ValueError("RiskAnonymizer requires texts for pre_stream_anonymize.")

        record_names = kwargs.get("record_names")
        if not isinstance(record_names, list) or len(record_names) != len(texts_or_indices):
            raise ValueError("record_names must be provided and aligned with texts_or_indices")
        if not all(isinstance(name, str) for name in record_names):
            raise ValueError("record_names entries must be strings")
        
        prior_edits_list = kwargs.get("prior_edits_list")
        
        if risk_scores is not None:
            self.set_risk_scores(risk_scores, records=self.dataset_records or None)
        
        for idx, (name, text) in enumerate(zip(record_names, texts_or_indices)):
            precomputed = self._precomputed_scores_and_spans(name)
            if precomputed is None:
                _, spans = self._tokenize(text)
                scores = self._compute_scores(text, spans, name)
            else:
                scores, spans = precomputed
            text_hash = self.hash_text(text)
            self._scores_cache[text_hash] = (scores, spans)
            
            if prior_edits_list is not None and idx < len(prior_edits_list):
                edits = prior_edits_list[idx]
                if edits:
                    self._prior_edits_cache[text_hash] = edits

    def add_dataset_records(self, dataset_records: Sequence[DatasetRecord]) -> None:
        self.dataset_records = list(dataset_records)

    def set_risk_scores(
        self,
        risk_scores: Dict[str, Dict[str, object]],
        records: Optional[Sequence[DatasetRecord]] = None,
    ) -> None:
        self._risk_scores_by_uid = {}
        self._risk_offsets_by_uid = {}
        self._risk_scores_ordered_by_uid = {}
        resolved = align_precomputed_risk_scores(risk_scores, records=records)
        for uid, entry in resolved.items():
            self._risk_scores_by_uid[uid] = entry.span_scores
            self._risk_offsets_by_uid[uid] = entry.ordered_offsets
            self._risk_scores_ordered_by_uid[uid] = entry.ordered_scores

    def _make_apply_fn(
        self,
        spans: List[Tuple[int, int]],
        runtime_stats: Dict[str, int],
    ) -> ApplyFn:
        def apply_fn(idx: int, ledger: TokenLedger) -> None:
            if idx >= len(spans):
                return
            token_text = ledger.entry(idx).original_text
            if token_text.lower() in self._terms_to_ignore:
                return
            ledger.replace(idx, self._mask_text)
            runtime_stats["masked"] += 1

        return apply_fn

    def anonymize_any_text(self, text: str, *args, buckets: Optional[Buckets] = None, record_name: Optional[str] = None, prior_edits: Optional[List[Dict[str, object]]] = None, **kwargs) -> List[Tuple[Dict[str, Any], AnonymizationResult]]:
        if buckets is None:
            buckets = []
        text_hash = self.hash_text(text)
        cached = self._scores_cache.get(text_hash)
        if cached is None:
            precomputed = self._precomputed_scores_and_spans(record_name)
            if precomputed is None:
                _, spans = self._tokenize(text)
                scores = self._compute_scores(text, spans, record_name)
            else:
                scores, spans = precomputed
        else:
            scores, spans = cached
        
        if prior_edits is None:
            prior_edits = self._prior_edits_cache.get(text_hash)
        
        if len(buckets) != 1 or not isinstance(buckets[0], RhoParams):
            raise ValueError("RiskAnonymizer only supports RhoParams for grid anonymization.")
        
        rho_params: RhoParams = buckets[0]
        
        if self._unit is None:
            self._unit = ByRiskUnit(temperature=self._temperature)
        
        self._unit.set_thresholds(rho_params.values(), name="rho")
        self._unit.set_risk_scores(scores)
        
        runtime_stats: Dict[str, int] = {"masked": 0}

        apply_fn = self._make_apply_fn(spans, runtime_stats)
        
        outputs: List[Tuple[Dict[str, Any], AnonymizationResult]] = []
        
        for step in self._unit.anonymize(
            text,
            spans,
            apply_fn,
            prior_edits=prior_edits,
        ):
            rho = step.threshold
            hp: Dict[str, Any] = {"rho": rho}
            
            private_text = step.text
            ledger = step.ledger
            
            result_edits = ledger.result_edits_metadata()
            result_spans: List[TextAnnotation] = []
            for edit in result_edits:
                if edit.get("kind") != "replaced":
                    continue
                span = edit.get("span")
                if not span:
                    continue
                result_spans.append(
                    TextAnnotation(
                        start=span[0],
                        end=span[1],
                        label="risk",
                        text=str(edit.get("text", "")),
                        replacement=str(edit.get("replacement", "")),
                    )
                )
            
            metadata: Dict[str, Any] = {
                "method": "risk",
                "rho": rho,
                "removed_count": runtime_stats["masked"],
                "total_tokens": len(spans),
                **step.metadata,
            }
            token_edits = [TokenEdit.from_mapping(e) for e in result_edits]
            
            outputs.append((
                hp,
                AnonymizationResult(
                    text=private_text,
                    annotations=TextAnnotations(spans=result_spans, token_edits=token_edits),
                    metadata=metadata,
                ),
            ))
        
        if not outputs:
            outputs.append((
                {"rho": 1.0},
                AnonymizationResult(
                    text=text,
                    annotations=TextAnnotations(),
                    metadata={"method": "risk", "masked": 0},
                ),
            ))
        
        return outputs

    def anonymize_from_dataset(self, idx: int, *args, **kwargs) -> AnonymizationResult:
        raise NotImplementedError("Use anonymize_any_text for RiskAnonymizer.")
    
    def _tokenize(self, text: str) -> Tuple[List[str], List[Tuple[int, int]]]:
        spans_data = self._splitter.tokenize_with_spans(text)
        tokens = [token for _, _, token in spans_data]
        spans = [(start, end) for start, end, _ in spans_data]
        return tokens, spans
    
    def _compute_scores(self, text: str, spans: List[Tuple[int, int]], record_name: Optional[str]) -> np.ndarray:
        precomputed = self._lookup_precomputed_scores(spans, record_name)
        if precomputed is not None:
            return precomputed
        raise NotImplementedError("RiskAnonymizer requires precomputed risk scores for each record.")

    def _precomputed_scores_and_spans(self, record_name: Optional[str]) -> Optional[Tuple[np.ndarray, List[Tuple[int, int]]]]:
        if not record_name:
            return None
        spans = self._risk_offsets_by_uid.get(record_name)
        scores = self._risk_scores_ordered_by_uid.get(record_name)
        if spans is None or scores is None:
            return None
        if len(spans) != len(scores):
            raise ValueError(f"Risk scores length mismatch for record {record_name}")
        return scores, spans

    def _lookup_precomputed_scores(self, spans: List[Tuple[int, int]], record_name: Optional[str]) -> Optional[np.ndarray]:
        if record_name and record_name in self._risk_scores_by_uid:
            mapping = self._risk_scores_by_uid[record_name]
            scores: List[float] = []
            for start, end in spans:
                if (start, end) not in mapping:
                    print(f"Missing score for span ({start}, {end}) in record {record_name}")
                    return None
                scores.append(mapping[(start, end)])
            return np.asarray(scores, dtype=float)
        return None
