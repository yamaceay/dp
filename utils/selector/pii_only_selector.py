from typing import List, Union
from dp.loaders.base import TextAnnotation, DatasetRecord
from dp.utils.selector.base import TokenSelector


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
        pii_detector=None,
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
    
    def select(self, text: str) -> List[TextAnnotation]:
        if not text or not text.strip():
            return []
        
        temp_record = DatasetRecord(text=text)
        
        predictions = self.pii_detector.predict([temp_record])
        
        if not predictions or not predictions[0].spans:
            return []
        
        filtered_spans = []
        for span in predictions[0].spans:
            if span.confidence is None or span.confidence >= self.threshold:
                filtered_spans.append(span)
        
        return filtered_spans