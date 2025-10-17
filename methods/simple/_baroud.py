from dp.methods.anonymizer import AnonymizationResult
from dp.methods.simple import SimpleAnonymizer

class BaroudAnonymizer(SimpleAnonymizer):
    def anonymize(self, text: str, *args, **kwargs) -> AnonymizationResult:
        return AnonymizationResult(text="[BAROUD ANONYMIZED TEXT]")
    
    def anonymize_by_idx(self, idx: int, *args, **kwargs) -> AnonymizationResult:
        return AnonymizationResult(text=f"[BAROUD ANONYMIZED TEXT BY IDX {idx}]")