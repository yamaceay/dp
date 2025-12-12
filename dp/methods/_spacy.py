from typing import List, Optional, Tuple

from dp.methods.anonymizer import AnonymizationResult, Anonymizer
from dp.methods.constants import Buckets, buckets_to_dicts, BucketDict
from dp.loaders.base import TextAnnotation

spacy_models = ["en_core_web_sm", "en_core_web_lg"]

class SpacyAnonymizer(Anonymizer):
    MODEL_NAME = "spacy"
    def __init__(self, *args, **kwargs):
        super().__init__(*args, model=self.MODEL_NAME, **kwargs)

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

    def anonymize_any_text(self, text: str, labels: List[str] = None, *args, buckets: Buckets = [], **kwargs) -> List[Tuple[BucketDict, AnonymizationResult]]:
        entities = self._extract_entities(text, labels)
        hp = {} if not buckets else buckets_to_dicts(buckets)[0]
        return [(hp, self._anonymize_entities(text, entities))]

    def _extract_entities(self, text: str, labels: Optional[List[str]]) -> List[TextAnnotation]:
        doc = self._nlp(text or "")
        entities: List[TextAnnotation] = []
        for ent in doc.ents:
            if labels and ent.label_ not in labels:
                continue
            entities.append(
                TextAnnotation(
                    start=ent.start_char,
                    end=ent.end_char,
                    label=ent.label_,
                    text=ent.text,
                    annotator="spacy",
                )
            )
        return entities

    def _anonymize_entities(
        self,
        text: str,
        entities: List[TextAnnotation],
    ) -> AnonymizationResult:
        filtered = list(entities)
        out_parts: List[str] = []
        last = 0
        for ann in filtered:
            out_parts.append(text[last:ann.start])
            out_parts.append(f"[{ann.label}]")
            last = ann.end
        out_parts.append(text[last:])
        anonymized = "".join(out_parts)
        metadata = {"method": "spacy"}
        return AnonymizationResult(text=anonymized, spans=filtered, metadata=metadata)
