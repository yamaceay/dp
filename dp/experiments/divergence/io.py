from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from dp.loaders import DatasetRecord


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for index, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                items.append(json.loads(text))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {index} in {path}") from exc
    return items


def build_divergence_evaluation_inputs(
    records: List[DatasetRecord],
    sources: Dict[str, Path],
) -> Dict[str, Dict[str, Any]]:
    index_to_key = {idx: record.uid for idx, record in enumerate(records)}
    evaluations: Dict[str, Dict[str, Any]] = {}
    for name, path in sources.items():
        entries = read_jsonl(path)
        texts: Dict[str, str] = {}
        metadata: Dict[str, Dict[str, Any]] = {}
        for entry in entries:
            idx = entry.get("idx")
            text = entry.get("text", "")
            if idx is None:
                continue
            try:
                idx_value = int(idx)
            except (TypeError, ValueError):
                continue
            key = index_to_key.get(idx_value)
            if not key:
                continue
            if text:
                texts[key] = text
            row_metadata = entry.get("metadata")
            if isinstance(row_metadata, dict):
                metadata[key] = dict(row_metadata)
        evaluations[name] = {
            "texts": texts,
            "total": len(entries),
            "metadata": metadata,
        }
    return evaluations


def build_record_info(records: List[DatasetRecord]) -> Dict[str, Dict[str, Any]]:
    info: Dict[str, Dict[str, Any]] = {}
    for index, record in enumerate(records, start=1):
        metadata = record.metadata or {}
        info[record.uid] = {
            "index": metadata.get("record_index", index),
            "name": record.name,
        }
    return info


def build_original_texts(records: List[DatasetRecord]) -> Dict[str, str]:
    return {record.uid: record.text for record in records}
