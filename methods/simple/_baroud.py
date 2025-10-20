from typing import List

from dp.methods.anonymizer import AnonymizationResult
from dp.methods.simple import SimpleAnonymizer
from dp.loaders.base import TextAnnotation, DatasetRecord

class BaroudAnonymizer(SimpleAnonymizer):
    def __init__(self, *args, pii_annotator: str = None, pii_threshold: float = 0.5, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.pii_annotator = pii_annotator
        self.pii_threshold = pii_threshold
        self.pii_detector = None
        
        if self.pii_annotator:
            from dp.utils.pii_detector import PIIDetector
            self.pii_detector = PIIDetector(model_name=self.pii_annotator, use_chunking=False)
    
    def _get_category_mask(self, label: str) -> str:
        if not label:
            return "[MASK]"
        return f"[{label}]"

    def anonymize_batch(self, texts: List[str], *args, **kwargs) -> List[AnonymizationResult]:
        if not self.pii_detector:
            raise ValueError("PII detector is not configured for BaroudAnonymizer.")
        
        results = []
        
        all_records = [DatasetRecord(text=text) for text in texts]
        pii_annotations_list = self.pii_detector.predict(all_records)

        for text, pii_annotations in zip(texts, pii_annotations_list):
            filtered_annotations = []
            for ann in pii_annotations.spans:
                if ann.confidence >= self.pii_threshold:
                    filtered_annotations.append(ann)
            
            sorted_annotations = sorted(filtered_annotations, key=lambda x: x.start, reverse=True)
            
            anonymized_text = text
            for ann in sorted_annotations:
                if 0 <= ann.start < ann.end <= len(anonymized_text):
                    category_mask = self._get_category_mask(ann.label)
                    anonymized_text = anonymized_text[:ann.start] + category_mask + anonymized_text[ann.end:]
            
            result_spans = [
                TextAnnotation(
                    start=ann.start,
                    end=ann.end,
                    label=ann.label,
                    text=ann.text,
                    replacement=self._get_category_mask(ann.label),
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
            
            result = AnonymizationResult(
                text=anonymized_text, 
                spans=result_spans, 
                metadata=metadata,
            )
            results.append(result)
        return results

    def anonymize(self, text: str, *args, **kwargs) -> AnonymizationResult:
        return self.anonymize_batch([text], *args, **kwargs)[0]
    
    def anonymize_from_dataset(self, idx: int, *args, **kwargs) -> AnonymizationResult:
        raise NotImplementedError("Use anonymize with text for BaroudAnonymizer.")