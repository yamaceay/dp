from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterable, List, Optional

from dp.loaders.base import DatasetAdapter, DatasetRecord, TextAnnotation

class TabDatasetAdapter(DatasetAdapter):
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
            with self.data_in.open("r", encoding="utf-8") as handle:
                self._records: List[dict] = json.load(handle)
        except Exception as exc:
            raise RuntimeError(f"Failed to load TAB dataset from {self.data_in}") from exc

    def __len__(self) -> int:
        return len(self._records)

    def iter_records(self) -> Iterable[DatasetRecord]:
        base_iter = ((idx, row) for idx, row in enumerate(self._records))
        for idx, row in self._slice_records(base_iter):
            uid = str(row.get("doc_id", idx))
            text = row.get("text", "")
            annotations_raw = row.get("annotations")
            spans = self._read_annotations(annotations_raw)

            meta = row.get("meta") or {}
            name = meta.get("applicant", "")
            metadata = {
                "country": meta.get("countries"),
                "year": meta.get("year"),
                "legal_branch": meta.get("legal_branch"),
                "articles": meta.get("articles"),
            }

            yield DatasetRecord(
                text=text,
                uid=uid,
                name=name,
                spans=spans,
                metadata=metadata,
            )

    def _read_annotations(self, annotations_raw: Optional[List[dict]]) -> Optional[List[TextAnnotation]]:
        if not annotations_raw:
            return None
        annotations_processed = []
        for annotator, annotations_one_person in annotations_raw.items():
            entity_mentions = annotations_one_person.get("entity_mentions", [])
            if not entity_mentions:
                continue
            for mention in entity_mentions:
                annotation = TextAnnotation(
                    start=mention.get("start_offset"),
                    end=mention.get("end_offset"),
                    label=mention.get("entity_type"),
                    text=mention.get("span_text"),
                    annotator=annotator,
                    metadata=mention.get("metadata", {
                        "identifier_type": mention.get("identifier_type"),
                        "confidential_status": mention.get("confidential_status"),
                    }),
                )
                annotations_processed.append(annotation)
        return annotations_processed
