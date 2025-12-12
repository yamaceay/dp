from typing import List, Tuple
from dp.methods.anonymizer import AnonymizationResult, Anonymizer
from dp.methods.constants import Buckets, buckets_to_dicts, BucketDict
from dp.loaders.base import TextAnnotation

class PresidioAnonymizer(Anonymizer):
    MODEL_NAME = "presidio"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, model=self.MODEL_NAME, **kwargs)

        try:
            from presidio_analyzer import AnalyzerEngine
        except Exception:
            self._analyzer = None
            self._presidio_available = False
        else:
            try:
                self._analyzer = AnalyzerEngine()
                self._presidio_available = True
            except Exception:
                self._analyzer = None
                self._presidio_available = False

    def anonymize_any_text(self, text: str, *args, buckets: Buckets = [], **kwargs) -> List[Tuple[BucketDict, AnonymizationResult]]:
        if not getattr(self, "_presidio_available", False) or self._analyzer is None:
            hp = {} if not buckets else buckets_to_dicts(buckets)[0]
            return [(hp, AnonymizationResult(text="[PRESIDIO ANONYMIZED TEXT]"))]

        analyzer = self._analyzer
        results = analyzer.analyze(text=text, language="en")
        spans = []
        out_parts = []
        last = 0
        results_sorted = sorted(results, key=lambda r: r.start)
        for r in results_sorted:
            start = int(r.start)
            end = int(r.end)
            if start < last:
                continue
            spans.append(TextAnnotation(
                start=start,
                end=end,
                label=r.entity_type,
                text=text[start:end],
                annotator="presidio"
            ))
            out_parts.append(text[last:start])
            out_parts.append(f"[{r.entity_type}]")
            last = end
        out_parts.append(text[last:])
        anonymized = "".join(out_parts)
        metadata = {"method": "presidio"}
        hp = {} if not buckets else buckets_to_dicts(buckets)[0]
        return [(hp, AnonymizationResult(text=anonymized, spans=spans, metadata=metadata))]