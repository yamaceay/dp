from typing import Union, List
from ..anonymizer import AnonymizationResult
from ..k_anon import KAnonymizer

class PetreAnonymizer(KAnonymizer):
    def anonymize(self, text: str, k: Union[int, List[int]], *args, **kwargs) -> AnonymizationResult:
        return AnonymizationResult(text="[PETRE K-ANONYMIZED TEXT]")