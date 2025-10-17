from typing import Union, List
from dp.methods.anonymizer import AnonymizationResult
from dp.methods.k_anon import KAnonymizer

class PetreAnonymizer(KAnonymizer):
    def anonymize_from_dataset(self, idx: int, k: Union[int, List[int]], *args, **kwargs) -> AnonymizationResult:
        return AnonymizationResult(text=f"[PETRE K-ANONYMIZED TEXT BY IDX {idx} WITH K {k}]")