from typing import Union, List, Optional
import torch
from abc import abstractmethod
from dp.methods.anonymizer import Anonymizer, AnonymizationResult

class DPAnonymizer(Anonymizer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.device = self._resolve_device(kwargs.get("device", None))
        print("Initialized DPAnonymizer")

    @abstractmethod
    def batch_anonymize(self, text: str, epsilon: List[float], *args, **kwargs) -> List[AnonymizationResult]:
        raise NotImplementedError()
    
    def anonymize(self, text: str, epsilon: float, *args, **kwargs) -> AnonymizationResult:
        return self.batch_anonymize(text, epsilon=[epsilon], *args, **kwargs)[0]
    
    def anonymize_from_dataset(self, idx: int, *args, **kwargs) -> AnonymizationResult:
        raise NotImplementedError("Use anonymize with text for DPAnonymizer.")

    def _resolve_device(self, device: Optional[Union[str, int, torch.device]]) -> torch.device:
        """Resolve device specification to torch.device."""
        if isinstance(device, torch.device):
            return device
        if device is None or device == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return torch.device("mps")
            return torch.device("cpu")
        if isinstance(device, str):
            return torch.device(device)
        if isinstance(device, int):
            if device >= 0 and torch.cuda.is_available():
                return torch.device(f"cuda:{device}")
            return torch.device("cpu")
        return torch.device("cpu")