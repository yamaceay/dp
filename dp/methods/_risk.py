from typing import Any, Dict, List, Optional, Tuple, Union
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
from dp.utils.explainer.base import TokenExplainer


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
        self._scores_cache: Dict[str, Tuple[np.ndarray, List[Tuple[int, int]]]] = {}
        self._starting_indices_cache: Dict[str, List[int]] = {}
        self._prior_edits_cache: Dict[str, List[Dict[str, object]]] = {}

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
            self.set_risk_scores(risk_scores)
        
        for idx, (name, text) in enumerate(zip(record_names, texts_or_indices)):
            _, spans = self._tokenize(text)
            scores = self._compute_scores(text, spans, name)
            text_hash = self.hash_text(text)
            self._scores_cache[text_hash] = (scores, spans)
            
            if prior_edits_list is not None and idx < len(prior_edits_list):
                edits = prior_edits_list[idx]
                if edits:
                    self._prior_edits_cache[text_hash] = edits

    def set_risk_scores(self, risk_scores: Dict[str, Dict[str, object]]) -> None:
        self._risk_scores_by_uid = {}
        
        for uid, payload in risk_scores.items():
            offsets = payload.get("offsets")
            scores = payload.get("scores")
            if offsets is None or scores is None:
                continue
            
            span_map: Dict[Tuple[int, int], float] = {}
            for span, value in zip(offsets, scores):
                if len(span) < 2:
                    continue
                span_map[(int(span[0]), int(span[1]))] = float(value)
            
            if span_map:
                self._risk_scores_by_uid[uid] = span_map

    def _starting_replacements_for_indices(
        self,
        uid: Optional[str],
        offsets: List[Tuple[int, int]],
    ) -> Tuple[Dict[int, str], Dict[int, str]]:
        return self._starting_replacements_and_labels_for_indices(uid, offsets)

    def _make_apply_fn(
        self,
        spans: List[Tuple[int, int]],
        runtime_stats: Dict[str, int],
    ) -> ApplyFn:
        def apply_fn(idx: int, ledger: TokenLedger) -> None:
            if idx >= len(spans):
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
            _, spans = self._tokenize(text)
            scores = self._compute_scores(text, spans, record_name)
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
            
            result_spans: List[TextAnnotation] = []
            for idx in step.new_indices:
                start, end = spans[idx]
                original = text[start:end]
                result_spans.append(
                    TextAnnotation(
                        start=start,
                        end=end,
                        label="risk",
                        text=original,
                        replacement=self._mask_text,
                    )
                )
            
            metadata: Dict[str, Any] = {
                "method": "risk",
                "rho": rho,
                "removed_count": runtime_stats["masked"],
                "total_tokens": len(spans),
                **step.metadata,
            }
            token_edits = [TokenEdit.from_mapping(e) for e in ledger.edits_metadata()]
            
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
        if self._explainer is None:
            raise RuntimeError("RiskAnonymizer requires explainer")
        raw_scores = self._explainer.explain(text, spans)
        return np.asarray(raw_scores, dtype=float)

    def _lookup_precomputed_scores(self, spans: List[Tuple[int, int]], record_name: Optional[str]) -> Optional[np.ndarray]:
        if record_name and record_name in self._risk_scores_by_uid:
            mapping = self._risk_scores_by_uid[record_name]
            scores: List[float] = []
            for start, end in spans:
                if (start, end) not in mapping:
                    return None
                scores.append(mapping[(start, end)])
            return np.asarray(scores, dtype=float)
        return None
