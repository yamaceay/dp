"""MIMIC dataset adapter with flexible split discovery."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional

import csv

from dp.loaders.base import DatasetAdapter, DatasetRecord, TextAnnotation

class MIMICDatasetAdapter(DatasetAdapter):
    """Adapter for the MIMIC dataset."""

    def __init__(
        self,
        data: Optional[str] = None,
        data_in: Optional[str] = None,
        max_records: Optional[int] = None,
        start: Optional[int] = None,
        end: Optional[int] = None,
        step: Optional[int] = None,
    ):
        self.data_in = Path(data_in)
        self.max_records = max_records
        self.start = start
        self.end = end
        self.step = step

        try:
            with open(self.data_in, newline='', encoding='utf-8') as csvfile:
                reader = csv.DictReader(csvfile)
                data = [row for row in reader]
            self._data = data
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(f"Failed to load MIMIC dataset from {self.data_in}") from exc

    def __len__(self) -> int:
        return len(self._data)

    def iter_records(self) -> Iterable[DatasetRecord]:
        all_records = self._data
        for record in self._slice_records(all_records):
            if any(k not in record for k in ["ROW_ID", "SUBJECT_ID", "TEXT"]):
                continue
            uid = str(record["ROW_ID"])
            name = str(record.get("SUBJECT_ID"))
            text = record.get("TEXT", "")
            spans: List[TextAnnotation] = []
            raw_spans_str = record.get("SPANS", "")
            try:
                raw_spans = eval(raw_spans_str)
            except Exception:
                raw_spans = []
            for span in raw_spans:
                start = int(span["start"])
                end = int(span["end"])
                label = span["label"].upper()
                replacement = text[start:end]
                spans.append(TextAnnotation(
                    start=start,
                    end=end,
                    label=label,
                    replacement=replacement,
                ))
            metadata = {
                "category": record.get("CATEGORY", ""),
                "description": record.get("DESCRIPTION", ""),
                "cgid": record.get("CGID", ""),
                "hadm_id": record.get("HADM_ID", ""),
                "chartdate": record.get("CHARTDATE", ""),
                "charttime": record.get("CHARTTIME", ""),
                "storetime": record.get("STORETIME", ""),
                "is_error": record.get("ISERROR", ""),
            }
            yield DatasetRecord(
                uid=uid, 
                name=name, 
                text=text, 
                metadata=metadata, 
                spans=spans
            )

__all__ = ["MIMICDatasetAdapter"]