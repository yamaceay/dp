from __future__ import annotations

from typing import Iterable, Iterator, List, Sequence

from dp.methods.anonymizer import AnonymizationResult, Anonymizer


class SimpleAnonymizer(Anonymizer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def add_dataset_records(self, dataset_records: Iterable) -> None:
        raise NotImplementedError("Simple methods operate directly on provided texts.")

    def anonymize(self, text: str, *args, **kwargs) -> AnonymizationResult:
        del args, kwargs
        return AnonymizationResult(text="[SIMPLE ANONYMIZED TEXT]")

    def anonymize_from_dataset(self, idx: int, *args, **kwargs) -> AnonymizationResult:
        del args, kwargs
        return AnonymizationResult(text=f"[SIMPLE ANONYMIZED TEXT BY IDX {idx}]")

    def anonymize_batch(self, texts: Sequence[str], *args, **kwargs) -> List[AnonymizationResult]:
        return [self.anonymize(text, *args, **kwargs) for text in texts]

    def anonymize_from_dataset_batch(self, indices: Sequence[int], *args, **kwargs) -> List[AnonymizationResult]:
        return [self.anonymize_from_dataset(idx, *args, **kwargs) for idx in indices]

    def anonymize_stream(self, texts: Sequence[str], *args, **kwargs) -> Iterator[AnonymizationResult]:
        for text in texts:
            yield self.anonymize(text, *args, **kwargs)
