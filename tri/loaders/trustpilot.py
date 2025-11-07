from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Protocol, Tuple
import json
from tqdm import tqdm

from dp.loaders import TrustpilotDatasetAdapter
from dp.tri.loaders.base import AttackerDatasetAdapter, AttackerDatasetRecord

class TrustpilotAttackerDatasetAdapter(AttackerDatasetAdapter):
    def __init__(
        self,
        *args,
        data: str | None = None,
        data_in: str | None = None,
        max_records: int | None = None,
        **kwargs,
    ) -> None:
        adapter = TrustpilotDatasetAdapter(data=data, data_in=data_in, max_records=max_records)
        super().__init__(adapter=adapter, *args, **kwargs)

    def extract_background_knowledge(self, record: DatasetRecord) -> List[Tuple[str, str]]:
        background_knowledge = []
        if 'records' not in record.metadata:
            raise ValueError("Record metadata must contain 'records' for background knowledge extraction.")
        reviews = record.metadata['records']
        if not isinstance(reviews, list) or len(reviews) == 0:
            raise ValueError("'records' in metadata must be a non-empty list.")

        for review_record in reviews:
            if 'stars' not in review_record:
                raise ValueError("Each review record must contain 'stars' field.")
            stars = review_record['stars']
            if 'text' not in review_record:
                raise ValueError("Each review record must contain 'text' field.")
            text = review_record['text']
            # if 'title' not in review_record:
            #     raise ValueError("Each review record must contain 'title' field.")
            # title = review_record['title']
            review_str = f"A user gave a {stars}-star review: '{text}'"
            background_knowledge.append(('review', review_str))

        return background_knowledge

    def clean_metadata(self, record: DatasetRecord) -> Dict[str, Any]:
        if 'records' in record.metadata:
            del record.metadata['records']
        return record.metadata