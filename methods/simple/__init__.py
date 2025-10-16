from ..anonymizer import Anonymizer, AnonymizationResult

class SimpleAnonymizer(Anonymizer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        print("Initialized SimpleAnonymizer")

    def anonymize(self, text: str, *args, **kwargs) -> AnonymizationResult:
        # placeholder simple anonymization
        return AnonymizationResult(text="[SIMPLE ANONYMIZED TEXT]")