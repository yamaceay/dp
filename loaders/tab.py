"""TAB (court cases) dataset adapter."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, List, Optional

from .base import DatasetAdapter, DatasetRecord, TextAnnotation

class TabDatasetAdapter(DatasetAdapter):
    """Adapter for the TAB anonymisation dataset."""

    def __init__(self, path: Optional[str] = None, max_records: Optional[int] = None):
        self.path = Path(path)
        self.max_records = max_records
        try:
            with self.path.open("r", encoding="utf-8") as handle:
                self._records: List[dict] = json.load(handle)
        except Exception as exc:  # pragma: no cover - IO safety
            raise RuntimeError(f"Failed to load TAB dataset from {self.path}") from exc

    def __len__(self) -> int:
        return len(self._records)

    def iter_records(self) -> Iterable[DatasetRecord]:
        for idx, row in enumerate(self._records):
            if self.max_records is not None and idx >= self.max_records:
                break

            uid = str(row.get("doc_id", idx))
            text = row.get("text", "")
            annotations_raw = row.get("annotations")
            annotations = self._read_annotations(annotations_raw)
            
            utilities = {
                "country": row.get("meta", {}).get("countries"),
                "years": row.get("meta", {}).get("years"),
            }
            name = row.get("meta", {}).get("applicant", "")
            metadata = {
                "quality_checked": row.get("quality_checked"),
                "task": row.get("task"),
                "dataset_type": row.get("dataset_type"),
                "meta": row.get("meta"),
            }

            yield DatasetRecord(
                uid=uid,
                text=text,
                name=name,
                annotations=annotations,
                utilities=utilities,
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