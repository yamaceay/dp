import string
from typing import List, Tuple
from dp.loaders.base import TextAnnotation
from dp.utils.selector.base import TokenSelector

class ByRiskSelector(TokenSelector):
    def __init__(
        self,
        risk_tolerance: float,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.set_risk_tolerance(risk_tolerance)

    def set_risk_tolerance(self, tolerance: float) -> None:
        value = float(tolerance)
        if value < 0 or value > 1:
            raise ValueError("risk_tolerance must be within [0, 1]")
        self._tolerance = value

    def select(self, text: str, offsets: List[Tuple[int, int]], risks: List[float]) -> List[TextAnnotation]:
        if not text or not text.strip():
            return []

        if len(risks) != len(offsets):
            return []

        removal_limit = max(0.0, min(1.0, 1.0 - self._tolerance))
        if removal_limit <= 0:
            return []

        filtered_spans: List[TextAnnotation] = []
        pairs = sorted(zip(offsets, risks), key=lambda item: item[1], reverse=True)
        removed = 0.0
        for offset, risk in pairs:
            if removed >= removal_limit:
                break
            contribution = float(risk) if risk is not None else 0.0
            if contribution <= 0:
                continue
            token = text[offset[0]:offset[1]]
            if not token or token in string.punctuation:
                continue
            filtered_spans.append(TextAnnotation(*offset, text=token))
            removed += contribution
        return filtered_spans
