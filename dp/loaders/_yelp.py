from __future__ import annotations

import json
from typing import Iterable

from dp.loaders.base import DatasetAdapter, DatasetRecord


class YelpDatasetAdapter(DatasetAdapter):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        with open(self.data_in, "r", encoding="utf-8") as f:
            self._records = [json.loads(line) for line in f if line.strip()]

    def __len__(self) -> int:
        return len(self._records)
    
    def iter_records(self) -> Iterable[DatasetRecord]:
        base_iter = ((idx, row) for idx, row in enumerate(self._records))
        for _, row in self._slice_records(base_iter):
            if "review_id" not in row or "user_id" not in row or "text" not in row:
                continue
            text = row["text"]
            name = row["user_id"]
            uid = row["review_id"]
            metadata = {
                "useful": row.get("useful"),
                "funny": row.get("funny"),
                "cool": row.get("cool"),
                "date": row.get("date"),
                "stars": row.get("stars"),
                "business_id": row.get("business_id"),
            }
            yield DatasetRecord(
                text=text,
                uid=uid,
                name=name,
                metadata=metadata,
            )

__all__ = ["YelpDatasetAdapter"]
