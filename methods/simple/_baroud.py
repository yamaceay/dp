from dp.methods.anonymizer import AnonymizationResult
from dp.methods.simple import SimpleAnonymizer
from dp.loaders.base import TextAnnotation, DatasetRecord

class BaroudAnonymizer(SimpleAnonymizer):
    def __init__(self, *args, pii_annotator: str = None, pii_threshold: float = 0.5, mask_token: str = "[MASKED]", **kwargs):
        super().__init__(*args, **kwargs)
        
        self.pii_annotator = pii_annotator
        self.pii_threshold = pii_threshold
        self.mask_token = mask_token
        self.pii_detector = None
        
        if self.pii_annotator:
            from dp.utils.pii_detector import PIIDetector
            self.pii_detector = PIIDetector(model_name=self.pii_annotator, use_chunking=False)

    def anonymize(self, text: str, *args, **kwargs) -> AnonymizationResult:
        if not text or not text.strip():
            return AnonymizationResult(
                text="",
                spans=[],
                metadata={"method": "baroud", "pii_detected": 0}
            )
        
        if not self.pii_detector:
            return AnonymizationResult(
                text=text,
                spans=[],
                metadata={"method": "baroud", "pii_detected": 0, "note": "no PII detector configured"}
            )
        
        pii_input = DatasetRecord(text=text)
        pii_annotations = self.pii_detector.predict([pii_input])[0]
        
        filtered_annotations = []
        for ann in pii_annotations.spans:
            if ann.confidence >= self.pii_threshold:
                filtered_annotations.append(ann)
        
        sorted_annotations = sorted(filtered_annotations, key=lambda x: x.start, reverse=True)
        
        anonymized_text = text
        for ann in sorted_annotations:
            if 0 <= ann.start < ann.end <= len(anonymized_text):
                anonymized_text = anonymized_text[:ann.start] + self.mask_token + anonymized_text[ann.end:]
        
        result_spans = [
            TextAnnotation(
                start=ann.start,
                end=ann.end,
                label=ann.label,
                text=ann.text,
                replacement=self.mask_token,
                confidence=ann.confidence,
                annotator="baroud"
            )
            for ann in filtered_annotations
        ]
        
        metadata = {
            "method": "baroud",
            "pii_detected": len(filtered_annotations),
            "threshold": self.pii_threshold,
        }
        
        return AnonymizationResult(text=anonymized_text, spans=result_spans, metadata=metadata)
    
    def anonymize_from_dataset(self, idx: int, *args, **kwargs) -> AnonymizationResult:
        raise NotImplementedError("Use anonymize with text for BaroudAnonymizer.")