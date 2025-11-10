from typing import List

from dp.methods.anonymizer import AnonymizationResult
from dp.methods.simple import SimpleAnonymizer
from dp.loaders.base import TextAnnotation

spacy_models = ["en_core_web_sm", "en_core_web_lg"]

class SpacyAnonymizer(SimpleAnonymizer):
    def __init__(self, *args, classification_threshold: float = 0.0, **kwargs):
        super().__init__(*args, **kwargs)
        self._pii_confidence = 0.0
        self.set_pii_confidence(classification_threshold)

        try:
            import spacy
        except Exception:
            raise ImportError("spaCy is not installed. Please install it with 'pip install spacy'.")

        if not hasattr(self, '_nlp'):
            model_loaded = False
            for model in spacy_models:
                try:
                    self._nlp = spacy.load(model)
                    model_loaded = True
                    break
                except Exception:
                    continue

            if not model_loaded:
                raise ImportError("Could not load any spaCy model. Please install one of: " + ", ".join(spacy_models))
                

    def set_pii_confidence(self, threshold: float) -> None:
        value = float(threshold)
        if value < 0 or value > 1:
            raise ValueError("pii_confidence must be between 0 and 1")
        self._pii_confidence = value

    def set_classification_threshold(self, threshold: float) -> None:
        self.set_pii_confidence(threshold)

    def _entity_confidence(self, entity) -> float:
        extension = getattr(entity, "_", None)
        if extension is not None and hasattr(extension, "get"):
            value = extension.get("confidence", None)
            if value is not None:
                try:
                    return float(value)
                except (TypeError, ValueError):
                    pass
        kb_value = getattr(entity, "kb_id_", None)
        if kb_value not in (None, ""):
            try:
                return float(kb_value)
            except (TypeError, ValueError):
                return 1.0
        return 1.0

    def anonymize(self, text: str, labels: List[str] = None, *args, **kwargs) -> AnonymizationResult:

        nlp = self._nlp

        doc = nlp(text or "")
        spans = []
        out_parts = []
        last = 0
        threshold = self._pii_confidence
        for ent in doc.ents:
            score = self._entity_confidence(ent)
            if score < threshold:
                continue
            if labels and ent.label_ not in labels:
                continue
            spans.append(TextAnnotation(
                start=ent.start_char,
                end=ent.end_char,
                label=ent.label_,
                text=ent.text,
                annotator="spacy"
            ))
            out_parts.append(text[last:ent.start_char])
            out_parts.append(f"[{ent.label_}]")
            last = ent.end_char
        out_parts.append(text[last:])
        anonymized = "".join(out_parts)
        metadata = {"method": "spacy"}
        return AnonymizationResult(text=anonymized, spans=spans, metadata=metadata)
    
    def anonymize_from_dataset(self, idx: int, *args, **kwargs) -> AnonymizationResult:
        raise NotImplementedError("Use anonymize with text for SpacyAnonymizer.")
