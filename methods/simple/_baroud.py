from ..anonymizer import AnonymizationResult
from ..simple import SimpleAnonymizer

class BaroudAnonymizer(SimpleAnonymizer):
    def anonymize(self, text: str, *args, **kwargs) -> AnonymizationResult:
        return AnonymizationResult(text="[BAROUD ANONYMIZED TEXT]")
