import string
from typing import Any, List, Optional, Tuple

from dp.loaders.base import TextAnnotation, DatasetRecord
from dp.utils.selector.base import AnonymizerUnit
from dp.utils.pii_detector import PIIDetector


class PIIOnlyUnit(AnonymizerUnit):
    def __init__(
        self,
        pii_detector: Optional[PIIDetector] = None,
        temperature: float = 1.0,
        sort_by_risk: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(temperature=temperature, sort_by_risk=sort_by_risk)
        if pii_detector is None:
            raise ValueError("PIIOnlyUnit requires a PIIDetector instance")
        self.pii_detector = pii_detector
        self._cached_predictions: Optional[List[TextAnnotation]] = None
        self._cached_text: Optional[str] = None

    def order_thresholds(self, thresholds: List[Any]) -> List[Any]:
        return sorted([float(t) for t in thresholds], reverse=True)

    def _get_predictions(self, text: str) -> List[TextAnnotation]:
        if self._cached_text == text and self._cached_predictions is not None:
            return self._cached_predictions
        temp_record = DatasetRecord(text=text)
        predictions = self.pii_detector.predict([temp_record])[0]
        spans = list(predictions.spans or []) if predictions else []
        self._cached_predictions = spans
        self._cached_text = text
        return spans

    def select_indices(
        self,
        text: str,
        offsets: List[Tuple[int, int]],
        threshold: Any,
        already_processed: set[int],
        **context: Any,
    ) -> List[int]:
        if not text or not text.strip():
            return []

        predictions = self._get_predictions(text)
        if not predictions:
            return []

        lambda_val = float(threshold)
        indices: List[int] = []

        for idx, (start, end) in enumerate(offsets):
            if idx in already_processed:
                continue
            token = text[start:end]
            if not token or token in string.punctuation:
                continue
            for span in predictions:
                if span.confidence is None or span.confidence < lambda_val:
                    continue
                if not (end <= span.start or start >= span.end):
                    indices.append(idx)
                    break

        if self._sort_by_risk_enabled:
            indices = self._sort_by_risk(indices, len(offsets))

        return indices