from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple
from visualize.common import (
    params_to_str_for_sort,
    read_jsonl_entries,
)

FILES: List[Tuple[str, str]] = [
    ("logs/reddit_priv_exp.jsonl", "reddit"),
    ("logs/tab_priv_exp.jsonl", "tab"),
]

def read_data(files: Sequence[Tuple[str, str]]) -> Iterable[Dict[str, Any]]:
    yield from read_jsonl_entries(files, kind="privacy")

def build_summaries() -> None:
    entries = list(read_data(FILES))
    entries = sorted(entries, key=lambda x: (x["method"], params_to_str_for_sort(x["params"])))
    by_dataset: Dict[str, List[Dict[str, Any]]] = {}
    for entry in entries:
        e = entry.copy()
        dataset = e.get("dataset")
        by_dataset.setdefault(dataset, []).append(e)
    Path("visualize/pretty").mkdir(parents=True, exist_ok=True)
    for dataset, dataset_entries in by_dataset.items():
        Path(f"visualize/pretty/{dataset}").mkdir(parents=True, exist_ok=True)
        with open(f"visualize/pretty/{dataset}/privacy.json", "w", encoding="utf-8") as f:
            json.dump(dataset_entries, f, indent=2)

if __name__ == "__main__":
    build_summaries()