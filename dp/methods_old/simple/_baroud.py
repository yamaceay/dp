from typing import Dict, List, Optional, Any

from dp.methods.anonymizer import AnonymizationResult
from dp.methods.simple import SimpleAnonymizer
from dp.loaders.base import TextAnnotation, DatasetRecord

class BaroudAnonymizer(SimpleAnonymizer):
    def __init__(self, *args, pii_annotator: str = None, pii_threshold: float = 0.5, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.pii_annotator = pii_annotator
        self.pii_threshold = 0.5
        self.pii_detector = None
        self.set_pii_confidence(pii_threshold)
        
        if self.pii_annotator:
            from dp.utils.pii_detector import PIIDetector
            self.pii_detector = PIIDetector(model_name=self.pii_annotator, use_chunking=False)
    
    def set_pii_confidence(self, threshold: float) -> None:
        value = float(threshold)
        if value < 0 or value > 1:
            raise ValueError("pii_confidence must be between 0 and 1")
        self.pii_threshold = value
    
    def set_classification_threshold(self, threshold: float) -> None:
        self.set_pii_confidence(threshold)
    
    def _get_category_mask(self, label: str) -> str:
        if not label:
            return "[MASK]"
        return f"[{label}]"

    def anonymize_batch(self, texts: List[str], *args, **kwargs) -> List[AnonymizationResult]:
        annotations = self._predict_pii_annotations(texts)
        results = []
        for text, anns in zip(texts, annotations):
            result = self._anonymize_from_annotations(text, anns, self.pii_threshold)
            results.append(result)
        return results

    def anonymize(self, text: str, *args, **kwargs) -> AnonymizationResult:
        return self.anonymize_batch([text], *args, **kwargs)[0]

    def anonymize_from_dataset(self, idx: int, *args, **kwargs) -> AnonymizationResult:
        raise NotImplementedError("Use anonymize with text for BaroudAnonymizer.")

    def grid_param_anonymize(
        self,
        *,
        param_name: str,
        values: List[float],
        texts: List[str],
        base_kwargs: Dict[str, Any],
        record_names: Optional[List[Optional[str]]],
        progress: bool,
    ) -> Optional[List[List[AnonymizationResult]]]:
        if param_name != "pii_confidence":
            return None
        annotations = self._predict_pii_annotations(texts)
        ordered_values = [float(value) for value in values]
        aggregated: List[List[AnonymizationResult]] = [[] for _ in texts]
        iterator = zip(texts, annotations)
        cached = list(iterator)
        for threshold in ordered_values:
            per_results = []
            for idx, (text, anns) in enumerate(cached):
                result = self._anonymize_from_annotations(text, anns, threshold)
                metadata = dict(result.metadata or {})
                metadata["pii_confidence"] = threshold
                metadata["_grid_param"] = "pii_confidence"
                metadata["_grid_value"] = threshold
                result.metadata = metadata
                per_results.append(result)
            for idx, result in enumerate(per_results):
                aggregated[idx].append(result)
        return aggregated

    def _predict_pii_annotations(self, texts: List[str]) -> List[List[TextAnnotation]]:
        if not self.pii_detector:
            raise ValueError("PII detector is not configured for BaroudAnonymizer.")
        records = [DatasetRecord(text=text) for text in texts]
        predictions = self.pii_detector.predict(records)
        return [list(pred.spans or []) for pred in predictions]

    def _anonymize_from_annotations(
        self,
        text: str,
        annotations: List[TextAnnotation],
        threshold: float,
    ) -> AnonymizationResult:
        filtered_annotations = [ann for ann in annotations if ann.confidence is None or ann.confidence >= threshold]
        sorted_annotations = sorted(filtered_annotations, key=lambda x: x.start, reverse=True)
        anonymized_text = text
        for ann in sorted_annotations:
            if 0 <= ann.start < ann.end <= len(anonymized_text):
                anonymized_text = (
                    anonymized_text[:ann.start]
                    + self._get_category_mask(ann.label)
                    + anonymized_text[ann.end:]
                )
        result_spans = [
            TextAnnotation(
                start=ann.start,
                end=ann.end,
                label=ann.label,
                text=ann.text,
                replacement=self._get_category_mask(ann.label),
                confidence=ann.confidence,
                annotator="baroud",
            )
            for ann in filtered_annotations
        ]
        metadata = {
            "method": "baroud",
            "pii_detected": len(filtered_annotations),
            "pii_confidence": threshold,
        }
        return AnonymizationResult(text=anonymized_text, spans=result_spans, metadata=metadata)
