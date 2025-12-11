from typing import List, Optional, Union

from dp.loaders import TextAnnotation, get_adapter

from dp.methods.anonymizer import AnonymizationResult, Anonymizer
from dp.methods.constants import Buckets

class ManualAnonymizer(Anonymizer):
    MODEL_NAME = "manual"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, model=self.MODEL_NAME, **kwargs)
        print("Initialized ManualAnonymizer")
        self.texts: List[str] = []
        self.annotations: List[List[TextAnnotation]] = []

    def pre_stream_anonymize(self, *args, texts_or_indices: Union[List[str], List[int]] = None, **kwargs) -> AnonymizationResult:
        if not all(isinstance(i, str) for i in texts_or_indices):
            raise ValueError("ManualAnonymizer requires dataset indices for pre_stream_anonymize.")
        self.texts += [record.text for record in self._dataset_records]
        self.annotations += [self._deduplicate_annotations(record.spans) for record in self._dataset_records]

    def anonymize_from_dataset(self, idx: int, *args, buckets: Buckets = [], **kwargs) -> AnonymizationResult:
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

    def _deduplicate_annotations(self, annotations: Optional[List[TextAnnotation]]) -> List[TextAnnotation]:
        if not annotations:
            return []

        last_end = -1
        deduped = []
        
        sorted_anns = sorted(annotations, key=lambda x: x.start)
        
        for ann in sorted_anns:
            start, end = ann.start, ann.end
            if start >= last_end:
                deduped.append(ann)
                last_end = end

        return deduped
