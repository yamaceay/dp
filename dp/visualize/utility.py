from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple
from dp.visualize.common import (
    params_to_str_for_sort,
    read_jsonl_entries,
)

requested_files = [
    (f"logs/{dataset}_{feature}_exp.jsonl", dataset, feature)
    for dataset, values in {
        "tab": ["country", "year"],
        "reddit": ["age", "income_level", "relationship_status", "sex", "birth_city_country", "education", "occupation", "city_country"],
        "db_bio": ["label"]
    }.items()
    for feature in values
]
FILES = [file for file in requested_files if Path(file[0]).exists()]

def read_data(files: Sequence[Tuple[str, str, str]]) -> Iterable[Dict[str, Any]]:
    yield from read_jsonl_entries(files, kind="utility")

def build_summaries() -> None:
    entries = list(read_data(FILES))
    entries = sorted(entries, key=lambda x: (x["method"], params_to_str_for_sort(x["params"])))
    
    by_dataset: Dict[str, List[Dict[str, Any]]] = {}
    for entry in entries:
        dataset = entry.get("dataset")
        if dataset:
            by_dataset.setdefault(dataset, []).append(entry)
    
    for dataset, dataset_entries in by_dataset.items():
        output_dir = Path("visualize/pretty") / dataset
        output_dir.mkdir(parents=True, exist_ok=True)
        
        with (output_dir / "utility.json").open("w", encoding="utf-8") as f:
            json.dump(dataset_entries, f, indent=2)
        
        by_feature: Dict[str, List[Dict[str, Any]]] = {}
        for entry in dataset_entries:
            feature = entry.get("feature")
            if feature:
                by_feature.setdefault(feature, []).append(entry)
        
        feature_dir = output_dir / "utility"
        feature_dir.mkdir(parents=True, exist_ok=True)

        for feature, feature_entries in by_feature.items():
            with (feature_dir / f"{feature}.json").open("w", encoding="utf-8") as f:
                json.dump(feature_entries, f, indent=2)

if __name__ == "__main__":
    build_summaries()