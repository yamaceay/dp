from dp.methods.anonymizer import AnonymizationResult
from dp.methods.simple import SimpleAnonymizer

spacy_models = ["en_core_web_sm", "en_core_web_lg"]

class SpacyAnonymizer(SimpleAnonymizer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        try:
            import spacy
        except Exception:
            return AnonymizationResult(text="[SPACY ANONYMIZED TEXT]")

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
                

    def anonymize(self, text: str, *args, **kwargs) -> AnonymizationResult:

        nlp = self._nlp

        doc = nlp(text or "")
        spans = []
        out_parts = []
        last = 0
        for ent in doc.ents:
            spans.append({"start": ent.start_char, "end": ent.end_char, "label": ent.label_, "text": ent.text})
            out_parts.append(text[last:ent.start_char])
            out_parts.append(f"[{ent.label_}]")
            last = ent.end_char
        out_parts.append(text[last:])
        anonymized = "".join(out_parts)
        metadata = {"method": "spacy"}
        return AnonymizationResult(text=anonymized, spans=spans, metadata=metadata)
    
    def anonymize_from_dataset(self, idx: int, *args, **kwargs) -> AnonymizationResult:
        raise NotImplementedError("Use anonymize with text for SpacyAnonymizer.")