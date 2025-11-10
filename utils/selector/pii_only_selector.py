import string
from typing import List, Optional, Tuple
from dp.loaders.base import TextAnnotation, DatasetRecord
from dp.utils.selector.base import TokenSelector
from dp.utils.pii_detector import PIIDetector


class PIIOnlySelector(TokenSelector):
    """
    Selector that identifies PII spans in text using a PIIDetector.
    
    This selector uses a trained PII detection model to identify personally
    identifiable information spans in text. Only spans above a confidence
    threshold are returned.
    
    Usage:
        from dp.utils.pii_detector import PIIDetector
        from dp.utils.selector import PIIOnlySelector
        
        # Load or create a trained PIIDetector
        detector = PIIDetector(model_name="path/to/trained/model")
        
        # Create selector with threshold
        selector = PIIOnlySelector(pii_detector=detector, threshold=0.7)
        
        # Use in anonymizer
        anonymizer.set_filtering_strategy(selector)
        
        # Or use directly
        pii_spans = selector.select("John Smith lives in New York")
    
    Args:
        pii_detector: Trained PIIDetector instance
        threshold: Minimum confidence threshold for PII detection (0.0-1.0)
        **kwargs: Additional configuration parameters
    """
    
    def __init__(
        self,
        pii_detector: Optional[PIIDetector] =None,
        threshold: float = 0.5,
        **kwargs
    ):
        """
        Initialize PIIOnlySelector.
        
        Args:
            pii_detector: PIIDetector instance (required)
            threshold: Confidence threshold for PII detection (default: 0.5)
            **kwargs: Additional configuration parameters
        """
        super().__init__(**kwargs)
        
        if pii_detector is None:
            raise ValueError(
                "PIIOnlySelector requires a PIIDetector instance. "
                "Please provide a pii_detector during initialization."
            )
        
        self.pii_detector = pii_detector
        self.threshold = threshold

    def select(self, text: str, offsets: Optional[List[Tuple[int, int]]] = None, labels: Optional[List[str]] = None) -> List[TextAnnotation]:
        if not text or not text.strip():
            return []
        
        temp_record = DatasetRecord(text=text)
        
        predictions = self.pii_detector.predict([temp_record])[0]
        
        if not predictions or not predictions.spans:
            return []
        
        filtered_spans = []
        for span in predictions.spans:
            if span.confidence is not None:
                if span.confidence < self.threshold:
                    continue
                if labels and span.label not in labels:
                    continue
            if span.text in string.punctuation:
                continue
            filtered_spans.append(span)
        
        if offsets is None:
            return filtered_spans

        shifted_spans = []
        for span in filtered_spans:
            if any(
                not (span.end <= offset[0] or span.start >= offset[1])
                for offset in offsets
            ):
                shifted_spans.append(span)
        return shifted_spans