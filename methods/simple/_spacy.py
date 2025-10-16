from ..anonymizer import AnonymizationResult
from ..simple import SimpleAnonymizer

class SpacyAnonymizer(SimpleAnonymizer):
    def anonymize(self, text: str, *args, **kwargs) -> AnonymizationResult:
        return AnonymizationResult(text="[SPACY ANONYMIZED TEXT]")