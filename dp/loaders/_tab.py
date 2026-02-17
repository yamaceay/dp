from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterable, List, Optional

from dp.loaders.base import DatasetAdapter, DatasetRecord, TextAnnotation, load_split_indices

class TabDatasetAdapter(DatasetAdapter):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.data_in = Path(self.data_in)
        self._records = self._load_records(self.data_in)
        self._split_indices = load_split_indices(data_name=self.data, split=self.split)
        if self._split_indices is not None:
            for idx in self._split_indices:
                if idx >= len(self._records):
                    raise ValueError(
                        f"Split index {idx} out of range for TAB dataset (size={len(self._records)})"
                    )

    def _load_records(self, source: Path) -> List[dict]:
        if source.is_file():
            return self._read_json_file(source)
        if source.is_dir():
            ordered_names = ["echr_test.json", "echr_dev.json", "echr_train.json"]
            ordered_files = [source / name for name in ordered_names if (source / name).is_file()]
            fallback_files = sorted(
                p for p in source.glob("*.json")
                if p.name not in {f.name for f in ordered_files}
            )
            files = ordered_files + fallback_files
            if not files:
                raise ValueError(f"No TAB json files found in directory '{source}'")
            records: List[dict] = []
            for file in files:
                records.extend(self._read_json_file(file))
            return records
        raise ValueError(f"data_in path '{source}' is not a valid file or directory")

    def _read_json_file(self, path: Path) -> List[dict]:
        try:
            with path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except Exception as exc:
            raise RuntimeError(f"Failed to load TAB dataset from {path}") from exc
        if not isinstance(payload, list):
            raise ValueError(f"TAB file must contain a JSON array: {path}")
        return payload

    def __len__(self) -> int:
        if self._split_indices is not None:
            return len(self._split_indices)
        return len(self._records)

    def iter_records(self) -> Iterable[DatasetRecord]:
        if self._split_indices is None:
            base_iter = ((idx, row) for idx, row in enumerate(self._records))
        else:
            base_iter = ((idx, self._records[idx]) for idx in self._split_indices)
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
