from __future__ import annotations

import glob
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence

import yaml

GridKey = str


@dataclass
class RuntimeConfigSet:
    base_config: Dict[str, Any] = field(default_factory=dict)
    k_values: List[int] = field(default_factory=list)
    epsilon_value: float = None
    pii_confidence_values: List[float] = field(default_factory=list)
    risk_tolerance_values: List[float] = field(default_factory=list)
    sources: List[str] = field(default_factory=list)


def load_runtime_bundle(raw_inputs: Optional[Sequence[str]]) -> RuntimeConfigSet:
    paths = _resolve_runtime_paths(raw_inputs)
    if not paths:
        return RuntimeConfigSet()
    configs = [(path, _read_yaml(path)) for path in paths]
    return _aggregate_configs(configs)


def _resolve_runtime_paths(raw_inputs: Optional[Sequence[str]]) -> List[str]:
    if not raw_inputs:
        return []
    resolved: List[str] = []
    for entry in raw_inputs:
        matches = sorted(glob.glob(entry))
        if matches:
            resolved.extend(matches)
            continue
        candidate = Path(entry)
        if not candidate.exists():
            raise FileNotFoundError(f"Runtime config '{entry}' does not exist")
        if not candidate.is_file():
            raise ValueError(f"Runtime config '{entry}' is not a file")
        resolved.append(str(candidate))
    ordered: List[str] = []
    seen = set()
    for path in resolved:
        normalized = str(Path(path))
        if normalized in seen:
            continue
        seen.add(normalized)
        ordered.append(normalized)
    return ordered


def _read_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as reader:
        data = yaml.safe_load(reader) or {}
        if not isinstance(data, dict):
            raise ValueError(f"Runtime config '{path}' must contain a mapping at the root")
        return data


ALIAS_TO_CANONICAL = {
    "k": "k",
    "epsilon": "epsilon",
    "lambda": "pii_confidence",
    "pii_confidence": "pii_confidence",
    "rho": "risk_tolerance",
    "risk_tolerance": "risk_tolerance",
}

CASTERS: Dict[str, Callable[[Any], Any]] = {
    "k": int,
    "epsilon": float,
    "pii_confidence": float,
    "risk_tolerance": float,
}


def _aggregate_configs(configs: Iterable[tuple[str, Dict[str, Any]]]) -> RuntimeConfigSet:
    base: Dict[str, Any] = {}
    numeric_values: Dict[str, List[Any]] = {
        "k": [],
        "epsilon": [],
        "pii_confidence": [],
        "risk_tolerance": [],
    }
    sources: List[str] = []
    for path, payload in configs:
        sources.append(path)
        for key, value in payload.items():
            if key == "runtime":
                continue
            canonical = ALIAS_TO_CANONICAL.get(key)
            if canonical:
                caster = CASTERS[canonical]
                numeric_values[canonical] = _merge_numeric_list(
                    numeric_values[canonical],
                    value,
                    caster,
                    path,
                    key,
                )
                continue
            _merge_scalar(base, key, value, path)
    return RuntimeConfigSet(
        base_config=base,
        k_values=numeric_values["k"],
        epsilon_value=numeric_values["epsilon"][0] if numeric_values["epsilon"] else None,
        pii_confidence_values=numeric_values["pii_confidence"],
        risk_tolerance_values=numeric_values["risk_tolerance"],
        sources=sources,
    )


def _merge_scalar(container: Dict[str, Any], key: str, value: Any, source: str) -> None:
    if key not in container:
        container[key] = value
        return
    existing = container[key]
    if existing == value:
        return
    raise ValueError(f"Runtime setting '{key}' from '{source}' conflicts with previous value")


def _merge_numeric_list(
    current: List[Any],
    value: Any,
    caster: Callable[[Any], Any],
    source: str,
    key: GridKey,
) -> List[Any]:
    values = _normalize_value_list(value, source, key)
    merged = list(current)
    for entry in values:
        casted = caster(entry)
        if casted in merged:
            continue
        merged.append(casted)
    return merged


def _normalize_value_list(value: Any, source: str, key: GridKey) -> List[Any]:
    if value is None:
        raise ValueError(f"Runtime setting '{key}' in '{source}' cannot be null")
    if isinstance(value, list):
        if not value:
            raise ValueError(f"Runtime setting '{key}' in '{source}' cannot be empty")
        return value
    return [value]
