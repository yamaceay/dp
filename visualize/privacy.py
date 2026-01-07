from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple
from visualize.common import (
    params_to_str_for_sort,
    params_to_str,
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
    by_dataset_by_method: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
    for entry in entries:
        e = entry.copy()
        dataset = e.pop("dataset")
        method = e.pop("method")
        by_dataset_by_method.setdefault(dataset, {}).setdefault(method, []).append(e)
    by_dataset_by_params: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
    for entry in entries:
        e = entry.copy()
        dataset = e.pop("dataset")
        params = params_to_str(e.pop("params"))
        by_dataset_by_params.setdefault(dataset, {}).setdefault(params, []).append(e)
    Path("visualize/pretty").mkdir(parents=True, exist_ok=True)
    with open("visualize/pretty/privacy.json", "w", encoding="utf-8") as f:
        json.dump(entries, f, indent=2)
    for dataset in by_dataset_by_method:
        with open(f"visualize/pretty/privacy_by_method_{dataset}.json", "w", encoding="utf-8") as f:
            json.dump(by_dataset_by_method[dataset], f, indent=2)
    for dataset in by_dataset_by_params:
        with open(f"visualize/pretty/privacy_by_params_{dataset}.json", "w", encoding="utf-8") as f:
            json.dump(by_dataset_by_params[dataset], f, indent=2)

if __name__ == "__main__":
    build_summaries()