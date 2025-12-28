from typing import List, Optional, Tuple

from dp.loaders import TextAnnotation, get_adapter
from dp.loaders.base import TextAnnotations

from dp.methods.anonymizer import AnonymizationResult, Anonymizer
from dp.methods.constants import Buckets, BucketDict

class ManualAnonymizer(Anonymizer):
    MODEL_NAME = "manual"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, model=self.MODEL_NAME, **kwargs)
        print("Initialized ManualAnonymizer")
        self.texts: List[str] = []
        self.annotations: List[List[TextAnnotation]] = []

    def add_dataset_records(self, dataset_records) -> None:
        self.set_dataset_records(dataset_records)
        self.texts = [record.text for record in self._dataset_records]
        self.annotations = [self._deduplicate_annotations(record.spans) for record in self._dataset_records]

    def pre_stream_anonymize(self, *args, **kwargs) -> None:
        return

    def anonymize_from_dataset(self, idx: int, *args, buckets: Buckets = [], **kwargs) -> List[Tuple[BucketDict, AnonymizationResult]]:
        text = self.texts[idx]
        spans = self.annotations[idx]
        
        result_spans: List[TextAnnotation] = []
        out_parts: List[str] = []
        last = 0
        result_cursor = 0
        
        for annotation in spans:
            prefix = text[last:annotation.start]
            replacement = f"[{annotation.label}]"
            result_start = result_cursor + len(prefix)
            result_end = result_start + len(replacement)
            result_spans.append(TextAnnotation(
                start=result_start,
                end=result_end,
                label=annotation.label,
                text=text[annotation.start:annotation.end],
                replacement=replacement,
                annotator="manual",
            ))
            out_parts.append(prefix)
            out_parts.append(replacement)
            result_cursor = result_end
            last = annotation.end
        
        out_parts.append(text[last:])
        anonymized = "".join(out_parts)
        
        metadata = {"method": "manual"}
        return [
            (
                BucketDict(),
                AnonymizationResult(
                    text=anonymized,
                    annotations=TextAnnotations(spans=result_spans),
                    metadata=metadata,
                ),
            )
        ]

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
