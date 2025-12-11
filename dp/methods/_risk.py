from typing import Dict, List, Optional, Tuple, Union
from hashlib import sha256

import numpy as np

from dp.loaders import TextAnnotation
from dp.methods.anonymizer import AnonymizationResult, Anonymizer
from dp.methods.constants import Buckets, RhoParams
from dp.utils.splitter import TextSplitter
from dp.utils.token_ledger import TokenLedger


class RiskAnonymizer(Anonymizer):
    MODEL_NAME = "risk"
    
    def __init__(self, *args, risk_temperature: float = 1.0, mask_text: str = "[MASK]", **kwargs):
        super().__init__(*args, model=self.MODEL_NAME, **kwargs)
        if risk_temperature <= 0:
            raise ValueError("temperature must be positive")
        self._mask_text = mask_text
        self._temperature = float(risk_temperature)
        self._splitter = TextSplitter()
        self._risk_scores_by_uid: Dict[str, Dict[Tuple[int, int], float]] = {}
        self._scores_cache: Dict[str, Tuple[np.ndarray, List[Tuple[int, int]]]] = {}
    
    def hash_text(self, text: str) -> str:
        return sha256(text.encode('utf-8')).hexdigest()
    
    def pre_stream_anonymize(self, texts_or_indices: Union[List[str], List[int]], *args, **kwargs) -> None:
        if not all(isinstance(i, str) for i in texts_or_indices):
            raise ValueError("RiskAnonymizer requires texts for pre_stream_anonymize.")
        
        risk_scores = kwargs.get('risk_scores')
        if risk_scores is not None:
            self.set_risk_scores(risk_scores)
        
        for text in texts_or_indices:
            tokens, spans = self._tokenize(text)
            scores = self._compute_scores(text, spans, kwargs.get('record_name'))
            self._scores_cache[self.hash_text(text)] = (scores, spans)

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

    def anonymize_any_text(self, text: str, *args, buckets: Buckets = [], record_name: Optional[str] = None, **kwargs) -> List[AnonymizationResult]:
        scores, spans = self._scores_cache[self.hash_text(text)]
        
        if len(buckets) != 1 or not isinstance(buckets[0], RhoParams):
            raise ValueError("RiskAnonymizer only supports RhoParams for grid anonymization.")
        
        rho_params: RhoParams = buckets[0]
        sorted_tolerances = sorted(rho_params.values(), reverse=True)
        
        ledger = TokenLedger(text, spans)
        aggregated: List[AnonymizationResult] = []
        
        probs = self._scores_to_probs(scores)
        ordered_indices = list(np.argsort(probs)[::-1])
        cumulative_probs = np.cumsum([float(probs[i]) for i in ordered_indices])
        
        masked_so_far = 0
        
        for tolerance in sorted_tolerances:
            removal_limit = max(0.0, min(1.0, 1.0 - tolerance))
            
            new_indices = []
            while masked_so_far < len(ordered_indices) and cumulative_probs[masked_so_far] <= removal_limit:
                idx = ordered_indices[masked_so_far]
                ledger.replace(idx, self._mask_text)
                new_indices.append(idx)
                masked_so_far += 1
            
            if new_indices:
                anonymized_text = ledger.render_offsets(text)
                result_spans = [
                    TextAnnotation(
                        start=spans[i][0],
                        end=spans[i][1],
                        label="risk",
                        text=text[spans[i][0]:spans[i][1]],
                        replacement=self._mask_text,
                    )
                    for i in new_indices
                ]
                metadata = {
                    "method": "risk",
                    "rho": tolerance,
                    "removed_count": len(new_indices),
                    "total_tokens": len(spans),
                }
                aggregated.append(AnonymizationResult(text=anonymized_text, spans=result_spans, metadata=metadata))
        
        return aggregated

    def anonymize_from_dataset(self, idx: int, *args, **kwargs) -> AnonymizationResult:
        raise NotImplementedError("Use anonymize_any_text for RiskAnonymizer.")
    
    def _tokenize(self, text: str) -> Tuple[List[str], List[Tuple[int, int]]]:
        spans_data = self._splitter.tokenize_with_spans(text)
        tokens = [token for _, _, token in spans_data]
        spans = [(start, end) for start, end, _ in spans_data]
        return tokens, spans
    
    def _compute_scores(self, text: str, spans: List[Tuple[int, int]], record_name: Optional[str]) -> np.ndarray:
        precomputed = self._lookup_precomputed_scores(text, spans, record_name)
        if precomputed is not None:
            return precomputed
        if self._explainer is None:
            raise RuntimeError("RiskAnonymizer requires explainer")
        raw_scores = self._explainer.explain(text, spans)
        return np.asarray(raw_scores, dtype=float)
    
    def _scores_to_probs(self, scores: np.ndarray) -> np.ndarray:
        if scores.size == 0:
            return scores
        scaled = scores / self._temperature
        scaled = scaled - np.max(scaled)
        exps = np.exp(scaled)
        total = np.sum(exps)
        if total <= 0:
            return np.ones(len(scores)) / len(scores)
        return exps / total
    
    def _select_tokens(self, probs: np.ndarray, tolerance: float) -> List[int]:
        removal_limit = max(0.0, min(1.0, 1.0 - tolerance))
        if removal_limit <= 0:
            return []
        ordered_indices = np.argsort(probs)[::-1]
        masked_indices = []
        removed = 0.0
        for idx in ordered_indices:
            if removed >= removal_limit:
                break
            masked_indices.append(int(idx))
            removed += float(probs[idx])
        return masked_indices

    def _lookup_precomputed_scores(self, text: str, spans: List[Tuple[int, int]], record_name: Optional[str]) -> Optional[np.ndarray]:
        if record_name and record_name in self._risk_scores_by_uid:
            mapping = self._risk_scores_by_uid[record_name]
            scores: List[float] = []
            for start, end in spans:
                if (start, end) not in mapping:
                    return None
                scores.append(mapping[(start, end)])
            return np.asarray(scores, dtype=float)
        return None
