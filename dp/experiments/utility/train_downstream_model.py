from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

from dp.loaders.base import DatasetRecord
from dp.loaders.derive import get_getter
from dp.tri import TRIDetector
from dp.tri.loaders import ATTACKER_ADAPTER_REGISTRY, AttackerDatasetRecord, get_attacker_adapter
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

available_datasets = list(ATTACKER_ADAPTER_REGISTRY.keys())


PROJECT_ROOT = Path(__file__).resolve().parent


def _read_yaml_mapping(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML root must be a mapping: {path}")
    return data


def _parse_scalar(raw: str) -> Any:
    value = raw.strip()
    lowered = value.lower()
    if lowered in {"null", "none"}:
        return None
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    try:
        if value.startswith("0") and len(value) > 1 and value[1].isdigit() and not value.startswith("0."):
            raise ValueError
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def _apply_set_overrides(cfg: dict[str, Any], overrides: Optional[list[str]]) -> None:
    if not overrides:
        return
    for spec in overrides:
        if not isinstance(spec, str) or "=" not in spec:
            raise ValueError(f"Invalid --set override (expected key=value): {spec!r}")
        key_raw, value_raw = spec.split("=", 1)
        key_raw = key_raw.strip()
        if not key_raw:
            raise ValueError(f"Invalid --set override (empty key): {spec!r}")
        value = _parse_scalar(value_raw)
        parts = [p.strip() for p in key_raw.split(".") if p.strip()]
        if not parts:
            raise ValueError(f"Invalid --set override (empty key path): {spec!r}")
        cur: Any = cfg
        for part in parts[:-1]:
            if not isinstance(cur, dict):
                raise ValueError(f"Invalid --set path (not a mapping at '{part}'): {key_raw}")
            nxt = cur.get(part)
            if nxt is None:
                nxt = {}
                cur[part] = nxt
            if not isinstance(nxt, dict):
                raise ValueError(f"Invalid --set path (existing non-mapping at '{part}'): {key_raw}")
            cur = nxt
        if not isinstance(cur, dict):
            raise ValueError(f"Invalid --set path (not a mapping): {key_raw}")
        cur[parts[-1]] = value


def _require_str(payload: dict[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Missing or invalid '{key}'")
    return value


def _optional_str(payload: dict[str, Any], key: str) -> Optional[str]:
    value = payload.get(key)
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Invalid '{key}'")
    return value


def _optional_int(payload: dict[str, Any], key: str) -> Optional[int]:
    value = payload.get(key)
    if value is None:
        return None
    if not isinstance(value, int):
        raise ValueError(f"Invalid '{key}'")
    return int(value)


def _optional_float(payload: dict[str, Any], key: str) -> Optional[float]:
    value = payload.get(key)
    if value is None:
        return None
    if not isinstance(value, (int, float)):
        raise ValueError(f"Invalid '{key}'")
    return float(value)


def _optional_bool(payload: dict[str, Any], key: str) -> Optional[bool]:
    value = payload.get(key)
    if value is None:
        return None
    if not isinstance(value, bool):
        raise ValueError(f"Invalid '{key}'")
    return bool(value)


def _get_mapping(payload: dict[str, Any], key: str) -> dict[str, Any]:
    value = payload.get(key)
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"Invalid '{key}'")
    return value


def _resolve_path(project_root: Path, raw: str) -> Path:
    p = Path(raw).expanduser()
    if not p.is_absolute():
        p = (project_root / p).resolve()
    return p


def _load_training_config(project_root: Path, config_path: Path) -> dict[str, Any]:
    resolved = config_path
    if not resolved.is_absolute():
        resolved = (project_root / resolved).resolve()
    payload = _read_yaml_mapping(resolved)

    dataset = payload.get("dataset")
    if not isinstance(dataset, str) or not dataset.strip():
        raise ValueError("Missing or invalid 'dataset'")
    if dataset not in available_datasets:
        raise ValueError(f"Unknown dataset: {dataset}")

    data_path = _resolve_path(project_root, _require_str(payload, "data_path"))
    model_name = _require_str(payload, "model_name")
    output_root = _resolve_path(project_root, _require_str(payload, "output_root"))
    training = _get_mapping(payload, "training")

    cfg: dict[str, Any] = {
        "dataset": dataset,
        "data_path": data_path,
        "attacker_extensions": _optional_str(payload, "attacker_extensions"),
        "model_name": model_name,
        "max_records": _optional_int(payload, "max_records"),
        "max_length": int(payload.get("max_length", 512)),
        "device": payload.get("device"),
        "output_root": output_root,
        "run_name": _optional_str(payload, "run_name"),
        "init_from": _optional_str(payload, "init_from"),
        "label_key": _optional_str(payload, "label_key"),
        "feature": _optional_str(payload, "feature"),
        "group_labels": bool(payload.get("group_labels", True)),
        "training": {
            "finetuning_epochs": int(training.get("finetuning_epochs", 15)),
            "batch_size": int(training.get("batch_size", 16)),
            "learning_rate": float(training.get("learning_rate", 5e-5)),
            "use_pretraining": bool(training.get("use_pretraining", False)),
            "pretraining_epochs": int(training.get("pretraining_epochs", 3)),
            "per_step": _optional_int(training, "per_step"),
            "early_stop_threshold": _optional_float(training, "early_stop_threshold"),
            "weight_decay": _optional_float(training, "weight_decay"),
            "warmup_ratio": _optional_float(training, "warmup_ratio"),
            "optimizer_type": _optional_str(training, "optimizer_type") or "adamw",
            "scheduler_type": _optional_str(training, "scheduler_type") or "constant",
        },
    }
    return cfg

def _freeze_encoder_only_classifier_train(model: AutoModelForSequenceClassification) -> None:
    for name, param in model.named_parameters():
        param.requires_grad = ("classifier" in name)

def _init_model_from_base(
    tri,
    base_path: str,
) -> None:
    tokenizer = AutoTokenizer.from_pretrained(base_path)
    model = AutoModelForSequenceClassification.from_pretrained(
        base_path,
        num_labels=tri.num_labels,
        problem_type="single_label_classification",
    )
    _freeze_encoder_only_classifier_train(model)
    device = tri.device if hasattr(tri, "device") else torch.device("cpu")
    model.to(device)
    tri.tokenizer = tokenizer
    tri.model = model

def _flatten_reddit_metadata(meta: Dict[str, Any], feature: Optional[str]) -> Dict[str, Any]:
    flat = dict(meta or {})
    if feature:
        flat["feature"] = feature
    persona = {}
    if isinstance(meta.get("persona"), dict):
        persona = dict(meta["persona"])  # type: ignore[index]
    if feature and feature in persona:
        flat[f"persona_{feature}"] = persona[feature]
    return flat

def _label_from_record(dataset: str, key: str, record: DatasetRecord, feature: Optional[str], group: bool) -> Optional[str]:
    if dataset == "reddit":
        tmp_meta = _flatten_reddit_metadata(record.metadata, feature)
        tmp = DatasetRecord(text=record.text, uid=record.uid, name=record.name, spans=record.spans, metadata=tmp_meta)
        getter_key = key if group else ("feature_label_exact" if key.startswith("feature_label") else key)
        getter = get_getter(dataset, getter_key)
        value = getter(tmp)
        return None if value is None else str(value)
    getter = get_getter(dataset, key)
    value = getter(record)
    return None if value is None else str(value)

def _inject_labels_as_names(
    records: List[AttackerDatasetRecord],
    dataset: str,
    label_key: str,
    feature: Optional[str],
    group_labels: bool,
) -> List[AttackerDatasetRecord]:
    out: List[AttackerDatasetRecord] = []
    for r in records:
        label = _label_from_record(dataset, label_key, r, feature, group_labels)
        if label is None or not str(label).strip():
            continue
        meta = dict(r.metadata or {})
        if dataset == "reddit" and feature:
            meta = _flatten_reddit_metadata(meta, feature)
        out.append(
            AttackerDatasetRecord(
                text=r.text,
                uid=r.uid,
                name=str(label),
                spans=r.spans,
                metadata=meta,
                train_texts=list(r.train_texts or []),
                eval_texts=list(r.eval_texts or []),
            )
        )
    return out

def main() -> int:
    parser = argparse.ArgumentParser(description="Train TRI downstream classifier with label injection")
    parser.add_argument("--mode", type=str, default="train", choices=["train", "evaluate", "predict"])

    parser.add_argument("--training-in", type=str, default=None)
    parser.add_argument("--output-root", type=str, default=None)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument(
        "--set",
        dest="set_overrides",
        action="append",
        default=None,
    )

    parser.add_argument("--dataset", type=str, default="tab", choices=available_datasets)
    parser.add_argument("--data_path", type=str, default=None)
    parser.add_argument("--model_name", type=str, default="distilbert-base-uncased")
    parser.add_argument("--max_records", type=int, default=None)
    parser.add_argument("--finetuning_epochs", type=int, default=15)
    parser.add_argument("--pretraining_epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--pretraining_batch_size", type=int, default=8)
    parser.add_argument("--use_pretraining", action="store_true")
    parser.add_argument("--per_step", type=int, default=None)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--attacker_extensions", type=str, default=None)
    parser.add_argument("--early_stop_threshold", type=float, default=None)
    parser.add_argument("--eval_on_original", action="store_true", help="Also evaluate on original record text (in addition to deidentified/rewritten)")
    parser.add_argument("--init_from", type=str, default=None, help="Path to base TRI checkpoint to reuse encoder weights from")
    parser.add_argument("--label_key", type=str, default=None, help="Label getter key from derive registry (e.g., 'feature_label', 'country', 'year')")
    parser.add_argument("--feature", type=str, default=None, help="Feature name for reddit when using feature-based keys (e.g., 'sex')")
    parser.add_argument("--group_labels", action="store_true", help="Group labels when supported by the getter (defaults to true in config mode)")

    args = parser.parse_args()

    if args.training_in is not None:
        cfg = _load_training_config(PROJECT_ROOT, Path(args.training_in))
        _apply_set_overrides(cfg, args.set_overrides)
        dataset = str(cfg["dataset"])
        data_path = Path(cfg["data_path"])
        model_name = str(cfg["model_name"])
        max_records = cfg.get("max_records")
        attacker_extensions = cfg.get("attacker_extensions")
        device = cfg.get("device")
        max_length = int(cfg.get("max_length", 512))
        training = cfg.get("training")
        if not isinstance(training, dict):
            raise SystemExit("Invalid training config")
        finetuning_epochs = int(training.get("finetuning_epochs", 15))
        batch_size = int(training.get("batch_size", 16))
        learning_rate = float(training.get("learning_rate", 5e-5))
        use_pretraining = bool(training.get("use_pretraining", False))
        pretraining_epochs = int(training.get("pretraining_epochs", 3))
        per_step = training.get("per_step")
        early_stop_threshold = training.get("early_stop_threshold")
        weight_decay = training.get("weight_decay")
        warmup_ratio = training.get("warmup_ratio")
        optimizer_type = str(training.get("optimizer_type", "adamw"))
        scheduler_type = str(training.get("scheduler_type", "constant"))
        init_from = cfg.get("init_from")
        label_key = cfg.get("label_key")
        feature = cfg.get("feature")
        group_labels = bool(cfg.get("group_labels", True))

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = args.run_name or cfg.get("run_name") or timestamp
        output_root = Path(args.output_root).expanduser().resolve() if args.output_root else Path(cfg["output_root"]).resolve()
        model_path: Path = (output_root / str(run_name)).resolve()
    else:
        if args.data_path is None:
            raise SystemExit("--data_path is required unless --training-in is provided")
        dataset = args.dataset
        data_path = Path(args.data_path)
        model_name = args.model_name
        max_records = args.max_records
        attacker_extensions = args.attacker_extensions
        device = args.device
        max_length = 512
        finetuning_epochs = args.finetuning_epochs
        batch_size = args.batch_size
        learning_rate = args.learning_rate
        use_pretraining = bool(args.use_pretraining)
        pretraining_epochs = args.pretraining_epochs
        per_step = args.per_step
        early_stop_threshold = args.early_stop_threshold
        init_from = args.init_from
        label_key = args.label_key
        feature = args.feature
        group_labels = bool(args.group_labels or False)
        weight_decay = None
        warmup_ratio = None
        optimizer_type = "adamw"
        scheduler_type = "constant"

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_root = Path(args.output_root).expanduser().resolve() if args.output_root else Path(f"models/tri_pipelines/{dataset}").resolve()
        run_name = args.run_name or timestamp
        model_path = Path(args.model_path).expanduser().resolve() if args.model_path else (output_root / run_name).resolve()

    adapter = get_attacker_adapter(dataset, data=dataset, data_in=str(data_path), max_records=max_records)
    if attacker_extensions:
        adapter.load_cache_from_jsonl(str(attacker_extensions))
    base_records: List[AttackerDatasetRecord] = list(adapter.iter_records())
    if not label_key:
        raise SystemExit("--label_key is required (or provide 'label_key' in training config)")
    records = _inject_labels_as_names(base_records, dataset, label_key, feature, group_labels)
    if not records:
        raise SystemExit("No records loaded")

    tri = TRIDetector(dataset_name=dataset, model_name=model_name, max_length=max_length, device=device)

    if args.mode == "train":
        tri.setup(records=records)
        if not init_from and args.model_path:
            init_from = args.model_path
        if init_from:
            base_path = Path(init_from).expanduser().resolve()
            if not base_path.exists():
                raise ValueError(f"Base checkpoint not found: {base_path}")
            _init_model_from_base(tri, str(base_path))
        else:
            tri.model_name = model_name
        model_path.mkdir(parents=True, exist_ok=True)
        tri.initialize_tokenizer_and_model()
        label_mapping_path = model_path / "label_mapping.json"
        with open(label_mapping_path, "w") as f:
            json.dump(tri.name_to_label, f, indent=2)
        tri.train(
            epochs=finetuning_epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            output_dir=str(model_path),
            use_pretraining=use_pretraining,
            pretraining_epochs=pretraining_epochs,
            early_stop_threshold=early_stop_threshold,
            per_step=per_step,
            weight_decay=weight_decay,
            warmup_ratio=warmup_ratio,
            optimizer_type=optimizer_type,
            scheduler_type=scheduler_type,
        )
        print(str(model_path))
        return 0

    if args.model_path is None:
        raise SystemExit("--model_path is required")
    tri.load(str(Path(args.model_path).expanduser().resolve()))
    tri.setup(records=records)

    if args.mode == "evaluate":
        results = tri.evaluate(tri.eval_records)
        print(results)
        return 0

    sample_records = tri.eval_records[:5] if len(tri.eval_records) >= 5 else tri.eval_records
    predictions = tri.predict(sample_records)
    print([{ "uid": uid, "top": max(probs.items(), key=lambda x: x[1])[0] } for uid, probs in list(predictions.items())[:5]])
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
