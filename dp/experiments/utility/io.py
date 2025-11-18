from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    entries: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                entries.append(json.loads(text))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number} in {path}") from exc
    return entries


def build_utility_evaluation_texts(
    index_to_key: Dict[int, str],
    sources: Dict[str, Path],
) -> Dict[str, Dict[str, str]]:
    evaluations: Dict[str, Dict[str, str]] = {}
    for name, path in sources.items():
        mapping: Dict[str, str] = {}
        for entry in read_jsonl(path):
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
            mapping[key] = text
        evaluations[name] = mapping
    return evaluations
