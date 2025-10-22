from typing import Dict, List, Optional, Sequence, Union
import torch
from abc import abstractmethod

from dp.methods.anonymizer import Anonymizer, AnonymizationResult


class DPAnonymizer(Anonymizer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.device = self._resolve_device(kwargs.get("device", None))
        print("Initialized DPAnonymizer")

    def add_dataset_records(self, dataset_records):
        raise NotImplementedError("DPAnonymizer does not support dataset records.")

    @abstractmethod
    def _grid_anonymize(
        self,
        text: str,
        epsilon: List[float],
        *args,
        **kwargs,
    ) -> Dict[float, List[AnonymizationResult]]:
        raise NotImplementedError()

    def batch_grid_anonymize(
        self,
        texts: Sequence[str],
        epsilon: List[float],
        *,
        progress: bool = False,
        **kwargs,
    ) -> Dict[float, List[List[AnonymizationResult]]]:
        if isinstance(texts, str):
            texts = [texts]
        else:
            texts = list(texts)
        if not texts:
            raise ValueError("texts cannot be empty")
        if isinstance(epsilon, (int, float)):
            ordered_eps = [float(epsilon)]
        else:
            ordered_eps = [float(e) for e in dict.fromkeys(epsilon)]
        aggregated: Dict[float, List[List[AnonymizationResult]]] = {value: [] for value in ordered_eps}
        texts_iter = texts
        if progress:
            from tqdm import tqdm
            texts_iter = tqdm(texts_iter, desc="DP batch anonymization")
        for text in texts_iter:
            per_text = self._grid_anonymize(text, ordered_eps, **kwargs)
            for eps_value, per_results in per_text.items():
                aggregated.setdefault(eps_value, []).append(per_results)
        return aggregated

    def batch_anonymize(
        self,
        texts: Sequence[str],
        epsilon: List[float],
        *,
        progress: bool = False,
        **kwargs,
    ) -> List[List[AnonymizationResult]] | Dict[float, List[List[AnonymizationResult]]]:
        if isinstance(texts, str):
            texts = [texts]
        else:
            texts = list(texts)
        if isinstance(epsilon, (int, float)):
            ordered_eps = [float(epsilon)]
        else:
            ordered_eps = [float(e) for e in dict.fromkeys(epsilon)]
        if not ordered_eps:
            raise ValueError("epsilon must contain at least one value")
        if len(ordered_eps) > 1:
            return self.batch_grid_anonymize(
                texts,
                ordered_eps,
                progress=progress,
                **kwargs,
            )
        single_eps = ordered_eps[0]
        aggregated: List[List[AnonymizationResult]] = []
        texts_iter = texts
        if progress:
            from tqdm import tqdm
            texts_iter = tqdm(texts_iter, desc="DP batch anonymization")
        for text in texts_iter:
            per_text = self._grid_anonymize(text, [single_eps], **kwargs)
            aggregated.append(per_text[single_eps])
        return aggregated

    def anonymize(
        self,
        text: str,
        epsilon: float,
        *args,
        **kwargs,
    ) -> AnonymizationResult:
        per_text = self._grid_anonymize(text, [epsilon], *args, **kwargs)
        results = per_text.get(epsilon)
        if not results:
            raise ValueError(f"No anonymization result produced for epsilon={epsilon}")
        return results[0]

    def anonymize_from_dataset(self, idx: int, *args, **kwargs) -> AnonymizationResult:
        raise NotImplementedError("Use anonymize with text for DPAnonymizer.")

    def _resolve_device(self, device: Optional[Union[str, int, torch.device]]) -> torch.device:
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
