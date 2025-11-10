from typing import Dict, List, Optional, Any

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
        if extension is not None and hasattr(extension, "confidence"):
            value = getattr(extension, "confidence")
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
        entities = self._extract_entities(text, labels)
        return self._anonymize_entities(text, entities, self._pii_confidence)

    def anonymize_from_dataset(self, idx: int, *args, **kwargs) -> AnonymizationResult:
        raise NotImplementedError("Use anonymize with text for SpacyAnonymizer.")

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
        texts_list = list(texts)
        entities_per_text = [self._extract_entities(text, base_kwargs.get("labels")) for text in texts_list]
        aggregated: List[List[AnonymizationResult]] = [[] for _ in texts_list]
        for threshold in values:
            per_results = []
            for text, entities in zip(texts_list, entities_per_text):
                result = self._anonymize_entities(text, entities, float(threshold))
                metadata = dict(result.metadata or {})
                metadata["pii_confidence"] = float(threshold)
                metadata["_grid_param"] = "pii_confidence"
                metadata["_grid_value"] = float(threshold)
                result.metadata = metadata
                per_results.append(result)
            for idx, result in enumerate(per_results):
                aggregated[idx].append(result)
        return aggregated

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
                    confidence=self._entity_confidence(ent),
                    annotator="spacy",
                )
            )
        return entities

    def _anonymize_entities(
        self,
        text: str,
        entities: List[TextAnnotation],
        threshold: float,
    ) -> AnonymizationResult:
        filtered = [ann for ann in entities if ann.confidence is None or ann.confidence >= threshold]
        out_parts: List[str] = []
        last = 0
        for ann in filtered:
            out_parts.append(text[last:ann.start])
            out_parts.append(f"[{ann.label}]")
            last = ann.end
        out_parts.append(text[last:])
        anonymized = "".join(out_parts)
        metadata = {"method": "spacy", "pii_confidence": threshold}
        return AnonymizationResult(text=anonymized, spans=filtered, metadata=metadata)
