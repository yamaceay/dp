"""Trustpilot dataset adapter."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional
import json

from datasets import load_dataset

from dp.loaders.base import DatasetAdapter, DatasetRecord
from dp.loaders.utils import recode_text

class TrustpilotDatasetAdapter(DatasetAdapter):
    """Adapter for Trustpilot review data."""

    def __init__(self, data: Optional[str], data_in: Optional[str] = None, max_records: Optional[int] = None):
        self.data_in = Path(data_in)
        self.max_records = max_records

        if data_in.endswith(".json") or data_in.endswith(".jsonl"):
            self._dataset = load_dataset("json", data_files={"data": str(self.data_in)})["data"]
        else:
            raise RuntimeError(f"Failed to load Trustpilot dataset from {self.data_in}")


    def __len__(self) -> int:
        return len(self._dataset)

    def iter_records(self) -> Iterable[DatasetRecord]:
        for idx, row in enumerate(self._dataset):
            if self.max_records is not None and idx >= self.max_records:
                break

            uid = str(row.get("review_id", idx))
            text = recode_text(row.get("review", ""))
            name = str(row.get("review_id", idx))
            utilities = {

            }
            metadata = {
                "category": row.get("category"),
                "stars": row.get("stars"),
                **dict(row)
            }

            yield DatasetRecord(
                text=text,
                uid=uid,
                name=name,
                metadata=metadata,
            )