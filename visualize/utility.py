from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple
from visualize.common import (
    params_to_str_for_sort,
    params_to_str,
    read_jsonl_entries,
)

FILES: List[Tuple[str, str, str]] = [
    ("logs/tab_country_exp.jsonl", "tab", "country"),
    ("logs/tab_year_exp.jsonl", "tab", "year"),
    ("logs/reddit_age_exp.jsonl", "reddit", "age"),
    ("logs/reddit_income_level_exp.jsonl", "reddit", "income_level"),
    ("logs/reddit_relationship_status_exp.jsonl", "reddit", "relationship_status"),
    ("logs/reddit_sex_exp.jsonl", "reddit", "sex"),
    ("logs/reddit_birth_city_country_exp.jsonl", "reddit", "birth_city_country"),
]

def read_data(files: Sequence[Tuple[str, str, str]]) -> Iterable[Dict[str, Any]]:
    yield from read_jsonl_entries(files, kind="utility")

def build_summaries() -> None:
    entries = list(read_data(FILES))
    entries = sorted(entries, key=lambda x: (x["method"], params_to_str_for_sort(x["params"])))
    by_dataset_by_feature_by_method: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
    for entry in entries:
        e = entry.copy()
        dataset = e.pop("dataset")
        method = e.pop("method")
        feature = e.pop("feature")
        by_dataset_by_feature_by_method.setdefault(dataset, {}).setdefault(feature, {}).setdefault(method, []).append(e)
    by_dataset_by_feature_by_params: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
    for entry in entries:
        e = entry.copy()
        dataset = e.pop("dataset")
        params = params_to_str(e.pop("params"))
        feature = e.pop("feature")
        by_dataset_by_feature_by_params.setdefault(dataset, {}).setdefault(feature, {}).setdefault(params, []).append(e)
    Path("visualize/pretty").mkdir(parents=True, exist_ok=True)
    with open("visualize/pretty/utility.json", "w", encoding="utf-8") as f:
        json.dump(entries, f, indent=2)
    for dataset in by_dataset_by_feature_by_method:
        for feature in by_dataset_by_feature_by_method[dataset]:
            with open(f"visualize/pretty/utility_by_method_{dataset}_{feature}.json", "w", encoding="utf-8") as f:
                json.dump(by_dataset_by_feature_by_method[dataset][feature], f, indent=2)
    for dataset in by_dataset_by_feature_by_params:
        for feature in by_dataset_by_feature_by_params[dataset]:
            with open(f"visualize/pretty/utility_by_params_{dataset}_{feature}.json", "w", encoding="utf-8") as f:
                json.dump(by_dataset_by_feature_by_params[dataset][feature], f, indent=2)

if __name__ == "__main__":
    build_summaries()