from ..anonymizer import AnonymizationResult
from ..simple import SimpleAnonymizer

class PresidioAnonymizer(SimpleAnonymizer):
    def anonymize(self, text: str, *args, **kwargs) -> AnonymizationResult:
        return AnonymizationResult(text="[PRESIDIO ANONYMIZED TEXT]")