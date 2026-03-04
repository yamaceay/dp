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


def build_privacy_evaluation_dataset_from_indexed_texts(
    records: Sequence[DatasetRecord],
    indexed_texts: Sequence[tuple[int | None, str]],
) -> List[DatasetRecord]:
    indexed_rows = sorted(
        [(index, text) for index, text in indexed_texts],
        key=lambda item: int(item[0]),
    )
    if len(indexed_rows) != len(records):
        raise ValueError(
            f"Indexed privacy texts count mismatch: got {len(indexed_rows)} rows, expected {len(records)}"
        )
    dataset: List[DatasetRecord] = []
    for expected_index, (maybe_index, text) in enumerate(indexed_rows):
        index = int(maybe_index)
        if index != expected_index:
            raise ValueError(
                f"Indexed privacy texts must be contiguous in [0, {len(records) - 1}]. "
                f"Expected idx={expected_index}, found idx={index}."
            )
        record = records[index]
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


def read_indexed_texts_from_jsonl(path: Path) -> List[tuple[int | None, str]]:
    rows: List[tuple[int, str]] = []
    seen_indices: set[int] = set()
    with path.open("r", encoding="utf-8") as reader:
        for line_number, raw in enumerate(reader, start=1):
            entry = raw.strip()
            if not entry:
                continue
            try:
                payload = json.loads(entry)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number} in {path}") from exc
            text = payload.get("text")
            if not isinstance(text, str):
                raise ValueError(f"Missing string 'text' on line {line_number} in {path}")
            idx_value = payload.get("idx")
            if idx_value is None:
                raise ValueError(f"Missing 'idx' on line {line_number} in {path}")
            try:
                index = int(idx_value)
            except Exception as exc:
                raise ValueError(f"Invalid integer 'idx' on line {line_number} in {path}: {idx_value}") from exc
            if index < 0:
                raise ValueError(f"Negative 'idx' on line {line_number} in {path}: {index}")
            if index in seen_indices:
                raise ValueError(f"Duplicate 'idx'={index} on line {line_number} in {path}")
            seen_indices.add(index)
            rows.append((index, text))
    return rows
