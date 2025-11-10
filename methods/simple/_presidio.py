from dp.methods.anonymizer import AnonymizationResult
from dp.methods.simple import SimpleAnonymizer
from dp.loaders.base import TextAnnotation

class PresidioAnonymizer(SimpleAnonymizer):
    def __init__(self, *args, classification_threshold: float = 0.0, **kwargs):
        super().__init__(*args, **kwargs)
        self._pii_confidence = 0.0
        self.set_pii_confidence(classification_threshold)
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

    def set_pii_confidence(self, threshold: float) -> None:
        value = float(threshold)
        if value < 0 or value > 1:
            raise ValueError("pii_confidence must be between 0 and 1")
        self._pii_confidence = value

    def set_classification_threshold(self, threshold: float) -> None:
        self.set_pii_confidence(threshold)

    def anonymize(self, text: str, *args, **kwargs) -> AnonymizationResult:
        if not getattr(self, "_presidio_available", False) or self._analyzer is None:
            return AnonymizationResult(text="[PRESIDIO ANONYMIZED TEXT]")

        analyzer = self._analyzer
        results = analyzer.analyze(text=text or "", language="en")
        spans = []
        out_parts = []
        last = 0
        results_sorted = sorted(results, key=lambda r: r.start)
        threshold = self._pii_confidence
        for r in results_sorted:
            start = int(r.start)
            end = int(r.end)
            if start < last:
                continue
            score = getattr(r, 'score', None)
            if score is None:
                score = 1.0
            if score < threshold:
                continue
            spans.append(TextAnnotation(
                start=start,
                end=end,
                label=r.entity_type,
                text=(text or "")[start:end],
                confidence=getattr(r, 'score', None),
                annotator="presidio"
            ))
            out_parts.append((text or "")[last:start])
            out_parts.append(f"[{r.entity_type}]")
            last = end
        out_parts.append((text or "")[last:])
        anonymized = "".join(out_parts)
        metadata = {"method": "presidio"}
        return AnonymizationResult(text=anonymized, spans=spans, metadata=metadata)
    
    def anonymize_from_dataset(self, idx: int, *args, **kwargs) -> AnonymizationResult:
        raise NotImplementedError("Use anonymize with text for PresidioAnonymizer.")
