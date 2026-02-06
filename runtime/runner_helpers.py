from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
import yaml

from dp.experiments.utility.io import build_utility_evaluation_texts
from dp.experiments.utility.models import UtilitySpec, resolve_model_choice
from dp.experiments.utility.vectorizer import (
    BERTVectorizer,
    SentenceEmbeddingVectorizer,
    SelfSupervisedFeatureExtractor,
    TfidfTextVectorizer,
)
from dp.loaders import DatasetRecord
from dp.loaders.derive import DERIVE_REGISTRY


DEFAULT_CONFIG_DIR = Path(__file__).resolve().parent.parent / "configs" / "experiments"


def ensure_sequence(obj: Any) -> List[Any]:
    if obj is None:
        return []
    if isinstance(obj, (list, tuple)):
        return list(obj)
    return [obj]


def resolve_config_path(value: str) -> Path:
    p = Path(value).expanduser()
    if p.is_file():
        return p
    candidate = (DEFAULT_CONFIG_DIR / value).resolve()
    if candidate.is_file():
        return candidate
    if not p.is_absolute():
        candidate = (DEFAULT_CONFIG_DIR / p.name).resolve()
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"Config file not found: {value}")


def load_config(path_value: Optional[str]) -> Dict[str, Any]:
    if not path_value:
        return {}
    resolved = resolve_config_path(path_value)
    with open(resolved, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def parse_component_config(payload: Any) -> Tuple[str, Dict[str, Any]]:
    if payload is None:
        return "", {}
    if isinstance(payload, str):
        return payload, {}
    if isinstance(payload, dict):
        cfg = dict(payload)
        name = str(cfg.pop("type", cfg.pop("name", "")))
        params = cfg.get("params", {}) if "params" in cfg else cfg
        if not isinstance(params, dict):
            raise ValueError("component params must be a mapping")
        return name, dict(params)
    raise ValueError("invalid component config")


def parse_metric_config(payload: Any) -> Tuple[str, Dict[str, Any]]:
    if payload is None:
        return "", {}
    if isinstance(payload, str):
        return payload, {}
    if isinstance(payload, dict):
        config = dict(payload)
        metric_type = str(config.pop("type", config.pop("name", "")))
        return metric_type, config
    raise ValueError("metric config must be a string or mapping")


def build_vectorizer_from_config(payload: Any) -> SelfSupervisedFeatureExtractor:
    name, params = parse_component_config(payload)
    if not name:
        name = "tfidf"
    name = name.lower()
    if name == "tfidf":
        return TfidfTextVectorizer(**params)
    if name == "bert":
        return BERTVectorizer(**params)
    if name in {"sentence", "sbert"}:
        return SentenceEmbeddingVectorizer(**params)
    raise ValueError(f"unsupported vectorizer '{name}'")


def select_records(records: List[DatasetRecord], criteria: Dict[str, Any]) -> List[DatasetRecord]:
    selected: List[DatasetRecord] = []
    for record in records:
        criteria_satisfied = True
        for key, value in criteria.items():
            got_value = None
            if key == "uid":
                got_value = record.uid
            elif key == "name":
                got_value = record.name
            elif key == "text":
                got_value = record.text
            else:
                got_value = record.metadata.get(key)
            if not isinstance(value, (list, tuple, set)):
                value = set([value])
            if isinstance(got_value, float):
                value = set(float(v) for v in value)
            elif isinstance(got_value, int):
                value = set(int(v) for v in value)
            else:
                value = set(str(v) for v in value)
            if got_value not in value:
                criteria_satisfied = False
                break
        if criteria_satisfied:
            selected.append(record)
    return selected


def align_evaluation_texts(records: List[DatasetRecord], sources: Dict[str, Path]) -> Dict[str, Dict[str, str]]:
    index_to_key = {idx: record.uid for idx, record in enumerate(records)}
    mapping = build_utility_evaluation_texts(index_to_key, sources)
    selected = {str(r.uid) for r in records}
    return {name: {k: v for k, v in m.items() if str(k) in selected} for name, m in mapping.items() if m}


def build_utility_target(params: Dict[str, Any], dataset: str) -> UtilitySpec:
    payload = params.get("target")
    top_level_preference = params.get("preference")
    if isinstance(payload, str):
        key = payload
        ttype = "nominal"
        enum: List[str] = []
        preference = str(top_level_preference).strip() if top_level_preference else None
    elif isinstance(payload, dict):
        key = str(payload.get("key", "")).strip()
        if not key:
            raise ValueError("target.key is required")
        ttype = str(payload.get("type", "nominal")).strip().lower() or "nominal"
        enum = ensure_sequence(payload.get("enum"))
        raw_pref = payload.get("preference", top_level_preference)
        preference = str(raw_pref).strip() if raw_pref else None
    else:
        raise ValueError("target must be string or mapping with key/type/enum")
    from dp.experiments.utility.base import UtilityTarget
    registry = DERIVE_REGISTRY.get(str(dataset)) or {}
    getter = registry.get(str(key))
    if getter is None:
        raise ValueError(f"unknown utility target key '{key}' for dataset '{dataset}'")
    try:
        mode = UtilityTarget.Mode(ttype)
    except Exception:
        raise ValueError("target.type must be one of: binary, nominal, ordinal, cardinal")
    if mode is UtilityTarget.Mode.ORDINAL and not enum:
        raise ValueError("ordinal target requires target.enum")
    if enum:
        allowed = set(enum)
        base_getter = getter
        def wrapped_getter(record: DatasetRecord) -> Any:
            v = base_getter(record)
            if v is None:
                return None
            s = str(v).strip()
            return s if s in allowed else None
        getter = wrapped_getter
    label_order = enum if mode is UtilityTarget.Mode.ORDINAL else None
    built = UtilityTarget(name=str(key), source=str(dataset), mode=mode, getter=getter, label_order=label_order)
    v_name, h_name = resolve_model_choice(mode, preference)
    return UtilitySpec(
        dataset=str(dataset),
        target_key=str(key),
        target=built,
        default_vectorizer=v_name,
        default_head=h_name,
        preference=preference,
    )
