from typing import Dict, Iterator, List, Optional, Sequence, Tuple, Union
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

    def _grid_anonymize(
        self,
        text: str,
        epsilon: List[float],
        *args,
        **kwargs,
    ) -> Dict[float, List[AnonymizationResult]]:
        ordered_eps = [float(e) for e in dict.fromkeys(epsilon)]
        if not ordered_eps:
            raise ValueError("epsilon must contain at least one value")
        aggregated: Dict[float, List[AnonymizationResult]] = {value: [] for value in ordered_eps}
        for eps_value, results in self._grid_anonymize_stream(
            text,
            ordered_eps,
            *args,
            **kwargs,
        ):
            if eps_value not in aggregated:
                aggregated[eps_value] = []
            aggregated[eps_value].extend(results)
        return aggregated

    @abstractmethod
    def _grid_anonymize_stream(
        self,
        text: str,
        epsilon: List[float],
        *args,
        **kwargs,
    ) -> Iterator[Tuple[float, List[AnonymizationResult]]]:
        raise NotImplementedError()

    def stream_batch_anonymize(
        self,
        texts: Sequence[str],
        epsilon: Sequence[float],
        *,
        progress: bool = False,
        **kwargs,
    ) -> Iterator[Dict[float, List[AnonymizationResult]]]:
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
        if not ordered_eps:
            raise ValueError("epsilon must contain at least one value")
        for batch in self._chunk_texts(texts, kwargs.pop("stream_batch_size", None)):
            for per_text in self._grid_anonymize_batch(
                batch,
                ordered_eps,
                progress=progress,
                **kwargs,
            ):
                yield per_text

    def anonymize_stream(
        self,
        texts: Sequence[str],
        epsilon: Sequence[float],
        *,
        progress: bool = False,
        **kwargs,
    ) -> Iterator[Dict[float, List[AnonymizationResult]]]:
        return self.stream_batch_anonymize(
            texts=texts,
            epsilon=epsilon,
            progress=progress,
            **kwargs,
        )

    def batch_grid_anonymize(
        self,
        texts: Sequence[str],
        epsilon: List[float],
        *,
        progress: bool = False,
        **kwargs,
    ) -> Dict[float, List[List[AnonymizationResult]]]:
        stream = self.stream_batch_anonymize(
            texts=texts,
            epsilon=epsilon,
            progress=progress,
            **kwargs,
        )
        if isinstance(epsilon, (int, float)):
            ordered_eps = [float(epsilon)]
        else:
            ordered_eps = [float(e) for e in dict.fromkeys(epsilon)]
        aggregated: Dict[float, List[List[AnonymizationResult]]] = {value: [] for value in ordered_eps}
        for per_text in stream:
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
        stream = self.stream_batch_anonymize(
            texts=texts,
            epsilon=epsilon,
            progress=progress,
            **kwargs,
        )
        if isinstance(epsilon, (int, float)):
            ordered_eps = [float(epsilon)]
        else:
            ordered_eps = [float(e) for e in dict.fromkeys(epsilon)]
        if len(ordered_eps) > 1:
            aggregated_dict: Dict[float, List[List[AnonymizationResult]]] = {value: [] for value in ordered_eps}
            for per_text in stream:
                for eps_value, per_results in per_text.items():
                    aggregated_dict.setdefault(eps_value, []).append(per_results)
            return aggregated_dict
        single_eps = ordered_eps[0]
        aggregated: List[List[AnonymizationResult]] = []
        for per_text in stream:
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

    def _grid_anonymize_batch(
        self,
        texts: Sequence[str],
        ordered_eps: List[float],
        *,
        progress: bool,
        **kwargs,
    ) -> Iterator[Dict[float, List[AnonymizationResult]]]:
        iterator = texts
        if progress:
            from tqdm import tqdm
            iterator = tqdm(iterator, desc="DP batch anonymization")
        for text in iterator:
            per_text: Dict[float, List[AnonymizationResult]] = {value: [] for value in ordered_eps}
            for eps_value, results in self._grid_anonymize_stream(
                text,
                ordered_eps,
                **kwargs,
            ):
                if eps_value not in per_text:
                    per_text[eps_value] = []
                per_text[eps_value].extend(results)
            yield per_text

    def _chunk_texts(self, texts: List[str], batch_size: Optional[int]) -> Iterator[List[str]]:
        if batch_size is None or batch_size <= 0:
            yield list(texts)
            return
        total = len(texts)
        for start in range(0, total, batch_size):
            yield texts[start:start + batch_size]
