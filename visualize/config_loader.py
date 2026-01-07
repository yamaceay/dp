from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import yaml

ALLOWED_DATASETS = {"reddit", "tab"}


def _load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError("config must be a mapping")
    return data


def list_method_sets(path: Path) -> List[str]:
    data = _load_yaml(path)
    methods = data.get("methods") or {}
    if not isinstance(methods, dict):
        raise ValueError("methods must be a mapping")
    return sorted(methods.keys())


def list_result_sets(path: Path) -> List[str]:
    data = _load_yaml(path)
    results = data.get("results") or {}
    if not isinstance(results, dict):
        raise ValueError("results must be a mapping")
    return sorted(results.keys())


def load_methods_set(path: Path, set_name: str) -> List[Any]:
    data = _load_yaml(path)
    methods = data.get("methods") or {}
    if set_name not in methods:
        raise KeyError(f"methods set not found: {set_name}")
    items = methods[set_name]
    if not isinstance(items, list):
        raise ValueError("methods set must be a list")
    return items


def load_results_set(path: Path, set_name: str) -> Tuple[str, str]:
    data = _load_yaml(path)
    results = data.get("results") or {}
    if set_name not in results:
        raise KeyError(f"results set not found: {set_name}")
    obj = results[set_name]
    if not isinstance(obj, dict):
        raise ValueError("results entry must be a mapping")
    dataset = str(obj.get("dataset")) if obj.get("dataset") is not None else ""
    flat = str(obj.get("flat")) if obj.get("flat") is not None else ""
    return dataset, flat


def validate_methods_config(path: Path) -> None:
    data = _load_yaml(path)
    version = data.get("version")
    if version not in {1, "1", None}:
        raise ValueError("unsupported methods config version")
    methods = data.get("methods")
    if not isinstance(methods, dict) or not methods:
        raise ValueError("methods must be a non-empty mapping")
    for set_name, items in methods.items():
        if not isinstance(items, list) or not items:
            raise ValueError(f"methods.{set_name} must be a non-empty list")
        for it in items:
            if isinstance(it, dict):
                if not it.get("method"):
                    raise ValueError(f"methods.{set_name} entry missing 'method'")
            elif not isinstance(it, str):
                raise ValueError(f"methods.{set_name} entries must be dict or str")


def validate_results_config(path: Path) -> None:
    data = _load_yaml(path)
    version = data.get("version")
    if version not in {1, "1", None}:
        raise ValueError("unsupported results config version")
    results = data.get("results")
    if not isinstance(results, dict) or not results:
        raise ValueError("results must be a non-empty mapping")
    for set_name, obj in results.items():
        if not isinstance(obj, dict):
            raise ValueError(f"results.{set_name} must be a mapping")
        dataset = obj.get("dataset")
        flat = obj.get("flat")
        if dataset not in ALLOWED_DATASETS:
            raise ValueError(f"results.{set_name}.dataset must be one of {sorted(ALLOWED_DATASETS)}")
        if not isinstance(flat, str) or not flat:
            raise ValueError(f"results.{set_name}.flat must be a non-empty string")
        
