from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Dict, Iterable, Optional, List
from datasets import load_dataset

from dp.loaders.base import DatasetAdapter, DatasetRecord

def recode_text(text: str) -> str:
    """Decode escape sequences in text while preserving Unicode."""
    replacements = {
        "\\n": "\n",
        "\\t": "\t",
        "\\r": "\r",
        '\\"': '"',
        "\\'": "'",
    }
    for escaped, replacement in replacements.items():
        text = text.replace(escaped, replacement)
    text = text.replace("\\\\", "\\")
    return text

class TrustpilotDatasetAdapter(DatasetAdapter):
    def __init__(
        self,
        data: Optional[str] = None,
        data_in: Optional[str] = None,
        max_records: Optional[int] = None,
        start: Optional[int] = None,
        end: Optional[int] = None,
        step: Optional[int] = None,
    ):
        if data_in is None:
            raise ValueError("data_in must point to a JSONL file")
        path = Path(data_in)
        if not path.exists():
            raise ValueError(f"Reddit dataset file not found: {path}")
        self.data_in = path
        self.max_records = max_records
        self.start = start
        self.end = end
        self.step = step
        self._records = list(self._read_records())

    def _read_records(self) -> Iterable[Dict]:
        dataset = load_dataset('csv', data_files={'train': str(self.data_in)})['train']
        for i, row in enumerate(dataset):
            yield {
                key: recode_text(value) 
                if isinstance(value, str) 
                else value
                for key, value in row.items()
            }

    def __len__(self) -> int:
        return len(self._records)

    def iter_records(self) -> Iterable[DatasetRecord]:
        base_iter = ((idx, row) for idx, row in enumerate(self._records))
        for idx, row in self._slice_records(base_iter):
            review_text = row.get('review')
            if review_text is None or not isinstance(review_text, str):
                raise ValueError("review must be a str")
            review_text = review_text.strip()
            if not review_text:
                continue
            company_name = row.get('company')
            if company_name is None or not isinstance(company_name, str):
                raise ValueError("company must be a str")
            company_name = company_name.strip()
            if not company_name:
                raise ValueError("company name cannot be empty")
            company_description = row.get('description')
            if company_description is None or not isinstance(company_description, str):
                raise ValueError("company_description must be a str")
            company_description = company_description.strip()
            if not company_description:
                raise ValueError("company description cannot be empty")
            review_title = row.get('title')
            if review_title is None or not isinstance(review_title, str):
                raise ValueError("title must be a str")
            review_title = review_title.strip()
            review_stars = row.get('stars')
            if review_stars is None or not isinstance(review_stars, int):
                raise ValueError("stars must be an int")
            category = row.get('category')
            if category is None or not isinstance(category, str):
                raise ValueError("category must be a str")
            category = category.strip()

            metadata = dict(
                review_stars=review_stars,
                review_title=review_title,
                category=category,
                company_description=company_description,
            )
            yield DatasetRecord(
                text=review_text,
                uid=f"tp_{idx + 1}",
                name=company_name,
                metadata=metadata,
            )

__all__ = ["TrustpilotDatasetAdapter"]