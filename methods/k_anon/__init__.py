from typing import Dict, Iterator, List, Optional, Tuple
from abc import abstractmethod

from dp.methods.anonymizer import Anonymizer, AnonymizationResult


class KAnonymizer(Anonymizer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        print("Initialized KAnonymizer")

    @abstractmethod
    def add_dataset_records(self, dataset_records):
        raise NotImplementedError()

    def _grid_anonymize_from_dataset(
        self,
        idx: int,
        k: List[int],
        **kwargs,
    ) -> Dict[int, List[AnonymizationResult]]:
        ordered_k = [int(value) for value in dict.fromkeys(k)]
        aggregated: Dict[int, List[AnonymizationResult]] = {value: [] for value in ordered_k}
        for k_value, results in self._grid_anonymize_stream_from_dataset(
            idx,
            ordered_k,
            **kwargs,
        ):
            if k_value not in aggregated:
                aggregated[k_value] = []
            aggregated[k_value].extend(results)
        return aggregated

    @abstractmethod
    def _grid_anonymize_stream_from_dataset(
        self,
        idx: int,
        k: List[int],
        **kwargs,
    ) -> Iterator[Tuple[int, List[AnonymizationResult]]]:
        raise NotImplementedError()

    def stream_batch_anonymize_from_dataset(
        self,
        indices: List[int],
        k: List[int],
        *,
        progress: bool = False,
        **kwargs,
    ) -> Iterator[Dict[int, List[AnonymizationResult]]]:
        if isinstance(indices, int):
            indices = [indices]
        else:
            indices = list(indices)
        if not indices:
            raise ValueError("indices cannot be empty")
        if isinstance(k, int):
            ordered_k = [k]
        else:
            ordered_k = list(dict.fromkeys(k))
        for batch in self._chunk_indices(indices, kwargs.pop("stream_batch_size", None)):
            for per_idx in self._grid_anonymize_batch_from_dataset(
                batch,
                ordered_k,
                progress=progress,
                **kwargs,
            ):
                yield per_idx

    def anonymize_stream(
        self,
        indices: List[int],
        k: List[int],
        *,
        progress: bool = False,
        **kwargs,
    ) -> Iterator[Dict[int, List[AnonymizationResult]]]:
        return self.stream_batch_anonymize_from_dataset(
            indices=indices,
            k=k,
            progress=progress,
            **kwargs,
        )

    def batch_grid_anonymize_from_dataset(
        self,
        indices: List[int],
        k: List[int],
        *,
        progress: bool = False,
        **kwargs,
    ) -> Dict[int, List[List[AnonymizationResult]]]:
        stream = self.stream_batch_anonymize_from_dataset(
            indices=indices,
            k=k,
            progress=progress,
            **kwargs,
        )
        if isinstance(k, int):
            ordered_k = [k]
        else:
            ordered_k = list(dict.fromkeys(k))
        aggregated: Dict[int, List[List[AnonymizationResult]]] = {value: [] for value in ordered_k}
        for per_idx in stream:
            for k_value, per_results in per_idx.items():
                aggregated.setdefault(k_value, []).append(per_results)
        return aggregated

    def batch_anonymize_from_dataset(
        self,
        indices: List[int],
        k: List[int],
        *,
        progress: bool = False,
        **kwargs,
    ) -> List[List[AnonymizationResult]] | Dict[int, List[List[AnonymizationResult]]]:
        stream = self.stream_batch_anonymize_from_dataset(
            indices=indices,
            k=k,
            progress=progress,
            **kwargs,
        )
        if isinstance(k, int):
            ordered_k = [k]
        else:
            ordered_k = list(dict.fromkeys(k))
        if len(ordered_k) > 1:
            aggregated_dict: Dict[int, List[List[AnonymizationResult]]] = {value: [] for value in ordered_k}
            for per_idx in stream:
                for k_value, per_results in per_idx.items():
                    aggregated_dict.setdefault(k_value, []).append(per_results)
            return aggregated_dict
        single_k = ordered_k[0]
        results: List[List[AnonymizationResult]] = []
        for per_idx in stream:
            results.append(per_idx[single_k])
        return results

    def anonymize(self, text: str, *args, **kwargs) -> AnonymizationResult:
        raise NotImplementedError("Use anonymize_from_dataset with dataset indices for KAnonymizer.")

    def anonymize_from_dataset(
        self,
        idx: int,
        k: int,
        *,
        progress: bool = False,
        **kwargs,
    ) -> AnonymizationResult:
        per_idx = self._grid_anonymize_from_dataset(idx, [k], **kwargs)
        results = per_idx.get(k)
        if not results:
            raise ValueError(f"No anonymization result produced for k={k}")
        return results[0]

    def _grid_anonymize_batch_from_dataset(
        self,
        indices: List[int],
        ordered_k: List[int],
        *,
        progress: bool,
        **kwargs,
    ) -> Iterator[Dict[int, List[AnonymizationResult]]]:
        iterator = indices
        if progress:
            from tqdm import tqdm
            iterator = tqdm(iterator, desc="K-anon batch anonymization")
        for idx in iterator:
            per_idx: Dict[int, List[AnonymizationResult]] = {value: [] for value in ordered_k}
            for k_value, results in self._grid_anonymize_stream_from_dataset(
                idx,
                ordered_k,
                **kwargs,
            ):
                if k_value not in per_idx:
                    per_idx[k_value] = []
                per_idx[k_value].extend(results)
            yield per_idx

    def _chunk_indices(self, indices: List[int], batch_size: Optional[int]) -> Iterator[List[int]]:
        if batch_size is None or batch_size <= 0:
            yield list(indices)
            return
        total = len(indices)
        for start in range(0, total, batch_size):
            yield indices[start:start + batch_size]
