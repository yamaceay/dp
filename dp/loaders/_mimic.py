from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

from dp.loaders.base import DatasetAdapter, DatasetRecord, load_split_indices


class MimicDatasetAdapter(DatasetAdapter):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._records: List[Tuple[str, str, int, str]] = list(self._read_flat_records())
        self._split_indices: Optional[List[int]] = load_split_indices(
            data_name=self.data, split=self.split
        )

    def _read_flat_records(self) -> Iterable[Tuple[str, str, int, str]]:
        with open(self.data_in, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                patient = json.loads(line)
                name = str(patient["name"])
                for idx, text in enumerate(patient.get("train_texts") or []):
                    yield ("train", name, idx, str(text))
                for idx, text in enumerate(patient.get("eval_texts") or []):
                    yield ("eval", name, idx, str(text))

    def __len__(self) -> int:
        if self._split_indices is not None:
            return len(self._split_indices)
        return len(self._records)

    def iter_records(self) -> Iterable[DatasetRecord]:
        if self._split_indices is not None:
            source = ((i, self._records[i]) for i in self._split_indices if i < len(self._records))
        else:
            source = enumerate(self._records)
        for _, (split, name, note_idx, text) in self._slice_records(source):
            yield DatasetRecord(
                text=text,
                uid=f"mimic_{name}_{split}_{note_idx}",
                name=name,
                metadata={"split": split, "note_idx": note_idx},
            )


__all__ = ["MimicDatasetAdapter"]
