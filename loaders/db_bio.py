"""DB-Bio dataset adapter with flexible split discovery."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Optional, Union

from datasets import Dataset, load_dataset

from .base import DatasetAdapter, DatasetRecord

class DBBioDatasetAdapter(DatasetAdapter):
    """Adapter for the DB-Bio legal dataset."""

    def __init__(
        self,
        data: Optional[str] = None,
        data_in: Optional[str] = None,
        max_records: Optional[int] = None,
    ):
        self.data_in = Path(data_in)
        self.max_records = max_records

        try:
            self._dataset = load_dataset("arrow", data_files={"data": str(self.data_in)})["data"]
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(f"Failed to load DB-Bio dataset from {self.data_in}") from exc

    def __len__(self) -> int:
        if isinstance(self._dataset, Dataset):
            return len(self._dataset)
        return len(self._dataset)

    def iter_records(self) -> Iterable[DatasetRecord]:
        if isinstance(self._dataset, Dataset):
            iterator = self._dataset
        else:
            iterator = self._dataset

        for idx, row in enumerate(iterator):
            if self.max_records is not None and idx >= self.max_records:
                break

            if isinstance(row, dict):
                data = row
            else:  # Dataset row returns dict already; safeguard
                data = dict(row)

            text = data.get("text", "")
            uid = data.get("wiki_name") or data.get("label") or str(idx)
            name = data.get("people")
            utilities = {
                "label": data.get("label"),
                "l1": data.get("l1"),
                "l2": data.get("l2"),
                "l3": data.get("l3"),
            }
            metadata = {
                "word_count": data.get("word_count"),
                "wiki_name": data.get("wiki_name"),
            }

            yield DatasetRecord(
                uid=str(uid),
                text=text,
                name=name,
                utilities=utilities,
                metadata=metadata,
            )

__all__ = ["DBBioDatasetAdapter"]
