from typing import Union, List
from dp.methods.anonymizer import AnonymizationResult
from dp.methods.dp import DPAnonymizer

class DPMlmAnonymizer(DPAnonymizer):
    def anonymize(self, text: str, epsilon: Union[float, List[float]], *args, **kwargs) -> AnonymizationResult:
        return AnonymizationResult(text="[DPMLM ANONYMIZED TEXT]")