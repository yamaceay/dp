from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from dp.loaders import DatasetRecord, TextAnnotation
from dp.methods.anonymizer import AnonymizationResult
from dp.methods.simple import SimpleAnonymizer
from dp.utils.splitter import TextSplitter


@dataclass(frozen=True)
class TokenRisk:
    token: str
    start: int
    end: int
    score: float
    probability: Optional[float]
    masked: bool


class RiskAnonymizer(SimpleAnonymizer):
    def __init__(
        self,
        *args,
        model: str,
        mask_text: str = "[MASK]",
        risk_threshold: float = 0.5,
        temperature: Optional[float] = None,
        tokenwise_epsilon_temperature: Optional[float] = None,
        **kwargs,
    ):
        if risk_threshold < 0:
            raise ValueError("risk_threshold must be non-negative")
        temp_value: Optional[float] = tokenwise_epsilon_temperature if tokenwise_epsilon_temperature is not None else temperature
        if temp_value is None:
            temp_value = 1.0
        if temp_value <= 0:
            raise ValueError("temperature must be positive")
        super().__init__(*args, model=model, **kwargs)
        self._mask_text = mask_text
        self._temperature = float(temp_value)
        self._threshold = float(risk_threshold)
        self._splitter = TextSplitter()
        self._dataset_records: List[DatasetRecord] = []
        self._explainer: Optional[object] = None

    def add_dataset_records(self, dataset_records: Iterable[DatasetRecord]) -> None:
        self._dataset_records.extend(dataset_records)

    def set_scoring_strategy(self, explainer) -> None:
        if not hasattr(explainer, "explain"):
            raise ValueError("Explainer must define explain")
        self._explainer = explainer

    def anonymize(self, text: str, *args, target_label: Optional[str] = None, record_name: Optional[str] = None, **kwargs) -> AnonymizationResult:
        tokens, spans = self._split_tokens(text)
        if not tokens:
            raise ValueError("tokenization produced no tokens")
        risk_scores, risk_probabilities, risk_source = self._score_tokens(text, tokens, target_label)
        risks = self._build_risks(tokens, spans, risk_scores, risk_probabilities)
        masked_text, mask_spans = self._apply_mask(text, risks)
        annotations = self._build_annotations(risks)
        metadata = self._build_metadata(
            record_name=record_name,
            target_label=target_label,
            risk_source=risk_source,
        )
        return AnonymizationResult(text=masked_text, spans=annotations, metadata=metadata)

    def anonymize_from_dataset(self, idx: int, *args, target_label: Optional[str] = None, **kwargs) -> AnonymizationResult:
        if idx < 0 or idx >= len(self._dataset_records):
            raise IndexError("dataset index out of range")
        record = self._dataset_records[idx]
        return self.anonymize(
            record.text,
            target_label=target_label,
            record_name=record.name or record.uid or None,
        )

    def _split_tokens(self, text: str) -> Tuple[List[str], List[Tuple[int, int]]]:
        spans = self._splitter.tokenize_with_spans(text)
        tokens = [token for _, _, token in spans]
        offsets = [(start, end) for start, end, _ in spans]
        return tokens, offsets

    def _score_tokens(
        self,
        text: str,
        tokens: Sequence[str],
        target_label: Optional[str],
    ) -> Tuple[np.ndarray, Optional[np.ndarray], str]:
        if self._explainer is None:
            raise RuntimeError("RiskAnonymizer requires set_scoring_strategy before use")
        raw_scores = self._explainer.explain(text, list(tokens), target_label=target_label)
        scores = np.asarray(raw_scores, dtype=float).ravel()
        probabilities = self._to_distribution(scores)
        source = type(self._explainer).__name__
        return scores, probabilities, source

    def _to_distribution(self, values: Optional[np.ndarray]) -> Optional[np.ndarray]:
        if values is None:
            return None
        if values.size == 0:
            return None
        array = np.asarray(values, dtype=float)
        finite_mask = np.isfinite(array)
        if not finite_mask.any():
            return None
        adjusted = array.copy()
        adjusted[~finite_mask] = float("-inf")
        scaled = adjusted / self._temperature
        scaled = scaled - np.max(scaled[np.isfinite(scaled)])
        exps = np.exp(scaled)
        total = np.sum(exps)
        if not np.isfinite(total) or total <= 0:
            length = array.shape[0]
            if length == 0:
                return None
            return np.full(length, 1.0 / length)
        return exps / total

    def _build_risks(
        self,
        tokens: Sequence[str],
        spans: Sequence[Tuple[int, int]],
        risk_scores: np.ndarray,
        risk_probabilities: Optional[np.ndarray],
    ) -> List[TokenRisk]:
        risks: List[TokenRisk] = []
        token_count = len(tokens)
        ordered_indices = sorted(range(token_count), key=lambda idx: risk_scores[idx], reverse=True)
        masked_flags = [False] * token_count

        cumulative_risk = 0.0
        for position, idx in enumerate(ordered_indices):
            if token_count == 0:
                break
            contribution: float
            if risk_probabilities is not None and idx < len(risk_probabilities):
                contribution = float(risk_probabilities[idx])
            else:
                contribution = float(risk_scores[idx]) if idx < len(risk_scores) else 0.0
            if not np.isfinite(contribution):
                contribution = 0.0
            contribution = max(contribution, 0.0)
            masked_flags[idx] = True
            cumulative_risk += contribution
            if cumulative_risk >= self._threshold:
                break

        for idx, token in enumerate(tokens):
            start, end = spans[idx] if idx < len(spans) else (0, 0)
            risk_score = float(risk_scores[idx]) if idx < len(risk_scores) else 0.0
            risk_probability = None
            if risk_probabilities is not None and idx < len(risk_probabilities):
                risk_probability = float(risk_probabilities[idx])
            risks.append(
                TokenRisk(
                    token=token,
                    start=start,
                    end=end,
                    score=risk_score,
                    probability=risk_probability,
                    masked=masked_flags[idx] and start < end,
                )
            )
        return risks

    def _apply_mask(self, text: str, risks: Sequence[TokenRisk]) -> Tuple[str, List[Tuple[int, int]]]:
        spans = [(risk.start, risk.end) for risk in risks if risk.masked]
        if not spans:
            return text, []
        masked = text
        for start, end in sorted(spans, key=lambda span: span[0], reverse=True):
            if start < 0 or end > len(masked) or start >= end:
                continue
            masked = masked[:start] + self._mask_text + masked[end:]
        return masked, spans

    def _build_annotations(self, risks: Sequence[TokenRisk]) -> List[TextAnnotation]:
        annotations: List[TextAnnotation] = []
        for risk in risks:
            if not risk.masked:
                continue
            annotations.append(
                TextAnnotation(
                    start=risk.start,
                    end=risk.end,
                    label=risk.token,
                    replacement=self._mask_text if risk.masked else risk.token,
                    metadata=self._risk_metadata(risk),
                )
            )
        return annotations

    def _risk_metadata(self, risk: TokenRisk) -> Dict[str, Optional[float]]:
        return {
            "risk_score": risk.score,
            "risk_probability": risk.probability,
            "masked": risk.masked,
        }

    def _build_metadata(
        self,
        record_name: Optional[str],
        target_label: Optional[str],
        risk_source: str,
    ) -> Dict[str, object]:
        return {
            "method": "risk",
            "record": record_name,
            "target_label": target_label,
            "threshold": self._threshold,
            "mask_text": self._mask_text,
            "risk_source": risk_source,
            "temperature": self._temperature,
        }
