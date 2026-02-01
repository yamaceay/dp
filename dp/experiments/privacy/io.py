from __future__ import annotations

from typing import List, Sequence
from pathlib import Path
import json

from dp.loaders import DatasetRecord, TextAnnotation
from dp.loaders.annotations import apply_annotations


def build_privacy_evaluation_dataset(
    records: List[DatasetRecord],
    annotations_batch: List[List[TextAnnotation]],
    mask_token: str,
) -> List[DatasetRecord]:
    dataset: List[DatasetRecord] = []
    total = len(annotations_batch)
    for index, record in enumerate(records):
        annotations = annotations_batch[index] if index < total else []
        text = apply_annotations(record.text, annotations, mask_text=mask_token)
        dataset.append(
            DatasetRecord(
                text=text,
                uid=record.uid,
                name=record.name,
                spans=record.spans,
                metadata=dict(record.metadata),
            )
        )
    return dataset


def build_privacy_evaluation_dataset_from_texts(
    records: Sequence[DatasetRecord],
    texts: Sequence[str],
) -> List[DatasetRecord]:
    total = min(len(records), len(texts))
    dataset: List[DatasetRecord] = []
    for index in range(total):
        record = records[index]
        dataset.append(
            DatasetRecord(
                text=str(texts[index]),
                uid=record.uid,
                name=record.name,
                spans=record.spans,
                metadata=dict(record.metadata),
            )
        )
    return dataset


def read_texts_from_jsonl(path: Path) -> List[str]:
    lines: List[str] = []
    with path.open("r", encoding="utf-8") as reader:
        for raw in reader:
            entry = raw.strip()
            if not entry:
                continue
            try:
                payload = json.loads(entry)
            except json.JSONDecodeError:
                continue
            text = payload.get("text")
            if isinstance(text, str):
                lines.append(text)
    return lines
