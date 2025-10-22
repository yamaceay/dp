from typing import Dict, List
from abc import abstractmethod

from dp.methods.anonymizer import Anonymizer, AnonymizationResult


class KAnonymizer(Anonymizer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        print("Initialized KAnonymizer")

    @abstractmethod
    def add_dataset_records(self, dataset_records):
        raise NotImplementedError()

    @abstractmethod
    def _grid_anonymize_from_dataset(
        self,
        idx: int,
        k: List[int],
        **kwargs,
    ) -> Dict[int, List[AnonymizationResult]]:
        raise NotImplementedError()

    def batch_grid_anonymize_from_dataset(
        self,
        indices: List[int],
        k: List[int],
        *,
        progress: bool = False,
        **kwargs,
    ) -> Dict[int, List[List[AnonymizationResult]]]:
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
        aggregated: Dict[int, List[List[AnonymizationResult]]] = {value: [] for value in ordered_k}
        iterator = indices
        if progress:
            from tqdm import tqdm
            iterator = tqdm(iterator, desc="K-anon batch anonymization")
        for idx in iterator:
            per_idx = self._grid_anonymize_from_dataset(idx, ordered_k, **kwargs)
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
        if isinstance(indices, int):
            indices = [indices]
        else:
            indices = list(indices)
        if isinstance(k, int):
            ordered_k = [k]
        else:
            ordered_k = list(dict.fromkeys(k))
        if not ordered_k:
            raise ValueError("k must contain at least one value")
        if len(ordered_k) > 1:
            return self.batch_grid_anonymize_from_dataset(
                indices,
                ordered_k,
                progress=progress,
                **kwargs,
            )
        single_k = ordered_k[0]
        results: List[List[AnonymizationResult]] = []
        iterator = indices
        if progress:
            from tqdm import tqdm
            iterator = tqdm(iterator, desc="K-anon batch anonymization")
        for idx in iterator:
            per_idx = self._grid_anonymize_from_dataset(idx, [single_k], **kwargs)
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
