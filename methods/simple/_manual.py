from typing import List, Optional

from dp.loaders import TextAnnotation, get_adapter

from dp.methods.anonymizer import AnonymizationResult
from dp.methods.simple import SimpleAnonymizer

class ManualAnonymizer(SimpleAnonymizer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        print("Initialized ManualAnonymizer")
        self.texts: List[str] = []
        self.annotations: List[List[TextAnnotation]] = []

    def add_dataset_records(self, dataset_records):
        self.texts += [record.text for record in dataset_records]
        self.annotations += [self._deduplicate_annotations(record.spans) for record in dataset_records]

    def anonymize(self, text: str, *args, **kwargs) -> AnonymizationResult:
        raise NotImplementedError("Use anonymize_from_dataset with an index for ManualAnonymizer.")

    def anonymize_from_dataset(self, idx: int, *args, **kwargs) -> AnonymizationResult:
        text = self.texts[idx]
        spans = self.annotations[idx]
        
        offset = 0
        for annotation in spans:
            start = annotation.start + offset
            end = annotation.end + offset
            replacement = f"[{annotation.label}]"
            
            text = text[:start] + replacement + text[end:]
            offset += len(replacement) - (end - start)
        
        metadata = {"method": "manual"}
        return AnonymizationResult(text=text, spans=spans, metadata=metadata)

    def _deduplicate_annotations(self, annotations: List[TextAnnotation]) -> List[TextAnnotation]:
        last_end = -1
        deduped = []
        
        sorted_anns = sorted(annotations, key=lambda x: x.start)
        
        for ann in sorted_anns:
            start, end = ann.start, ann.end
            if start >= last_end:
                deduped.append(ann)
                last_end = end

        return deduped