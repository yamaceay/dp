import string
from typing import List, Optional, Tuple
from dp.loaders.base import TextAnnotation
from dp.utils.selector.base import TokenSelector

class ByRiskSelector(TokenSelector):
    def __init__(
        self,
        threshold: float,
        **kwargs
    ):
        """
        Initialize ByRiskSelector.
        
        Args:
            threshold: Confidence threshold for risk detection (default: 0.5)
            **kwargs: Additional configuration parameters
        """
        super().__init__(**kwargs)
        self.threshold = threshold

    def select(self, text: str, offsets: List[Tuple[int, int]], risks: List[float]) -> List[TextAnnotation]:
        if not text or not text.strip():
            return []
        
        if not risks or len(risks) != len(offsets):
            return []

        filtered_spans = []
        for (offset, risk) in zip(offsets, risks):
            if risk < self.threshold:
                continue
            token = text[offset[0]:offset[1]]
            if token in string.punctuation:
                continue
            filtered_spans.append(TextAnnotation(*offset, text=token))

        return filtered_spans
