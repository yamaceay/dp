from dp.methods.anonymizer import Anonymizer, AnonymizationResult

class SimpleAnonymizer(Anonymizer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        print("Initialized SimpleAnonymizer")

    def anonymize(self, text: str, *args, **kwargs) -> AnonymizationResult:
        # placeholder simple anonymization

        text = "[SIMPLE ANONYMIZED TEXT]"
        return AnonymizationResult(text=text)

    def anonymize_from_dataset(self, idx: int, *args, **kwargs) -> AnonymizationResult:
        # placeholder simple anonymization by index
        
        text = f"[SIMPLE ANONYMIZED TEXT BY IDX {idx}]"
        return AnonymizationResult(text=text)