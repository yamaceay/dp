"""Trustpilot dataset adapter."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

from datasets import load_dataset

from .base import DatasetAdapter, DatasetRecord
from .utils import recode_text

class TrustpilotDatasetAdapter(DatasetAdapter):
    """Adapter for Trustpilot review data."""

    def __init__(self, path: Optional[str] = None, max_records: Optional[int] = None):
        self.path = Path(path)
        self.max_records = max_records

        try:
            self._dataset = load_dataset("json", data_files={"data": str(self.path)})["data"]
        except Exception as exc:  # pragma: no cover - import/runtime safety
            raise RuntimeError(f"Failed to load Trustpilot dataset from {self.path}") from exc

    def __len__(self) -> int:
        return len(self._dataset)

    def iter_records(self) -> Iterable[DatasetRecord]:
        for idx, row in enumerate(self._dataset):
            if self.max_records is not None and idx >= self.max_records:
                break

            uid = str(row.get("review_id", idx))
            text = recode_text(row.get("review", ""))
            utilities = {
                "category": row.get("category"),
                "stars": row.get("stars"),
            }
            metadata = dict(row)

            yield DatasetRecord(
                uid=uid,
                text=text,
                utilities=utilities,
                metadata=metadata,
            )