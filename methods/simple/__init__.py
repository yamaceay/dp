from typing import List
from dp.methods.anonymizer import Anonymizer, AnonymizationResult

class SimpleAnonymizer(Anonymizer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        print("Initialized SimpleAnonymizer")

    def add_dataset_records(self, dataset_records):
        raise NotImplementedError("SimpleAnonymizer does not support dataset records.")

    def anonymize(self, text: str, *args, **kwargs) -> AnonymizationResult:
        text = "[SIMPLE ANONYMIZED TEXT]"
        return AnonymizationResult(text=text)

    def anonymize_from_dataset(self, idx: int, *args, **kwargs) -> AnonymizationResult:
        text = f"[SIMPLE ANONYMIZED TEXT BY IDX {idx}]"
        return AnonymizationResult(text=text)
    
    def anonymize_batch(self, texts: List[str], *args, **kwargs) -> List[AnonymizationResult]:
        results = []
        for text in texts:
            result = self.anonymize(text, *args, **kwargs)
            results.append(result)
        return results
    
    def anonymize_from_dataset_batch(self, indices: List[int], *args, **kwargs) -> List[AnonymizationResult]:
        results = []
        for idx in indices:
            result = self.anonymize_from_dataset(idx, *args, **kwargs)
            results.append(result)
        return results