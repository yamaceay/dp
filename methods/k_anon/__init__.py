from typing import List
from abc import abstractmethod
from dp.methods.anonymizer import Anonymizer, AnonymizationResult


class KAnonymizer(Anonymizer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        print("Initialized KAnonymizer")

    @abstractmethod
    def batch_anonymize_from_dataset(self, idx: int, k: List[int], *args, **kwargs) -> List[AnonymizationResult]:
        raise NotImplementedError()

    def anonymize(self, text: str, *args, **kwargs) -> AnonymizationResult:
        raise NotImplementedError("Use anonymize_from_dataset with idx for KAnonymizer.")
    
    def anonymize_from_dataset(self, idx: int, k: int, *args, **kwargs) -> AnonymizationResult:
        return self.batch_anonymize_from_dataset(idx, k=[k], *args, **kwargs)[0]
    