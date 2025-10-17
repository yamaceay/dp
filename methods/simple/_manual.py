from typing import List, Optional

from dp.loaders import TextAnnotation, get_adapter

from dp.methods.anonymizer import AnonymizationResult
from dp.methods.simple import SimpleAnonymizer

class ManualAnonymizer(SimpleAnonymizer):
    def __init__(self, data: str = None, data_in: str = None, max_records: int = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        print("Initialized ManualAnonymizer")
        dataset = get_adapter(data, data_in=data_in, max_records=max_records)
        self.texts = [record.text for record in dataset]
        self.annotations = [self._deduplicate_annotations(record.annotations) for record in dataset]

    def anonymize(self, text: str, *args, **kwargs) -> AnonymizationResult:
        raise NotImplementedError("Use anonymize_from_dataset with an index for ManualAnonymizer.")

    def anonymize_from_dataset(self, idx: int, *args, **kwargs) -> AnonymizationResult:
        # Manual anonymization logic goes here

        text = self.texts[idx]
        for annotation in self.annotations[idx]:
            text = text.replace(annotation.text, f"[{annotation.label}]")

        return AnonymizationResult(text=text)
    
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