from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Set

import yaml


def _load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError("catalog must be a mapping")
    return data


def load_catalog(path: Path) -> Dict[str, Any]:
    return _load_yaml(path)


def _non_empty_string(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _unique(values: Iterable[str]) -> bool:
    seen: Set[str] = set()
    for v in values:
        if v in seen:
            return False
        seen.add(v)
    return True


def validate_catalog(catalog: Dict[str, Any]) -> None:
    version = catalog.get("version")
    if version not in {1, "1", None}:
        raise ValueError("unsupported catalog version")
    datasets = catalog.get("datasets") or {}
    if not isinstance(datasets, dict) or not datasets:
        raise ValueError("datasets must be a non-empty mapping")
    experiments = catalog.get("experiments") or []
    if not isinstance(experiments, list) or not experiments:
        raise ValueError("experiments must be a non-empty list")
    names: List[str] = []
    for exp in experiments:
        if not isinstance(exp, dict):
            raise ValueError("each experiment must be a mapping")
        name = exp.get("name")
        dataset = exp.get("dataset")
        method_class = exp.get("method_class")
        if not _non_empty_string(name):
            raise ValueError("experiment.name must be a non-empty string")
        if not _non_empty_string(dataset):
            raise ValueError(f"{name}: dataset must be a non-empty string")
        if dataset not in datasets:
            raise ValueError(f"{name}: dataset '{dataset}' not declared under datasets")
        if not _non_empty_string(method_class):
            raise ValueError(f"{name}: method_class must be a non-empty string")
        names.append(str(name))
    if not _unique(names):
        raise ValueError("experiment names must be unique")


def get_experiments(catalog: Dict[str, Any]) -> List[Dict[str, Any]]:
    experiments = catalog.get("experiments") or []
    return [e for e in experiments if isinstance(e, dict)]


def to_runtime_spec(exp: Dict[str, Any]) -> Dict[str, Any]:
    allow_keys = {
        "name",
        "dataset",
        "method_class",
        "selection_criteria",
        "target",
        "mode",
        "vectorizer",
        "head",
        "seed",
        "output",
    }
    return {k: v for k, v in exp.items() if k in allow_keys}


__all__ = [
    "_load_yaml",
    "get_experiments",
    "load_catalog",
    "to_runtime_spec",
    "validate_catalog",
]
