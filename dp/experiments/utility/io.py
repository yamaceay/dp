from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Set, Tuple


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    entries: List[Dict[str, Any]] = []
    for entry in iter_jsonl(path):
        entries.append(entry)
    return entries


def iter_jsonl(path: Path) -> Iterator[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number} in {path}") from exc
            if not isinstance(parsed, dict):
                continue
            yield parsed


def build_utility_evaluation_mapping(
    index_to_key: Dict[int, str],
    path: Path,
    selected_keys: Optional[Set[str]] = None,
) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    for entry in iter_jsonl(path):
        idx = entry.get("idx")
        text = entry.get("text", "")
        if idx is None or not text:
            continue
        try:
            idx_value = int(idx)
        except (TypeError, ValueError):
            continue
        key = index_to_key.get(idx_value)
        if not key:
            continue
        normalized_key = str(key)
        if selected_keys is not None and normalized_key not in selected_keys:
            continue
        mapping[normalized_key] = text
    return mapping


def iter_utility_evaluation_texts(
    index_to_key: Dict[int, str],
    sources: Dict[str, Path],
    selected_keys: Optional[Set[str]] = None,
) -> Iterator[Tuple[str, Dict[str, str]]]:
    for name, path in sorted(sources.items(), key=lambda item: item[0]):
        mapping = build_utility_evaluation_mapping(index_to_key, path, selected_keys=selected_keys)
        if mapping:
            yield name, mapping


def build_utility_evaluation_texts(
    index_to_key: Dict[int, str],
    sources: Dict[str, Path],
) -> Dict[str, Dict[str, str]]:
    evaluations: Dict[str, Dict[str, str]] = {}
    for name, path in sources.items():
        mapping = build_utility_evaluation_mapping(index_to_key, path)
        evaluations[name] = mapping
    return evaluations
