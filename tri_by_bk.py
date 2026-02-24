from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Any, Optional
import yaml

from dp.tri.base import TRIDetector
from dp.tri.loaders import get_attacker_adapter, ATTACKER_ADAPTER_REGISTRY
from dp.utils.tasking import resolve_task_id, apply_task_template


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


def _optional_int_or_none(payload: dict[str, Any], key: str) -> Optional[int]:
    value = payload.get(key)
    if value is None:
        return None
    if not isinstance(value, int):
        raise ValueError(f"Invalid '{key}'")
    return int(value)


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
    lr = training.get("lr", 5e-5)
    if not isinstance(lr, (int, float)):
        raise ValueError("Invalid 'training.lr'")
    scheduler_type = training.get("scheduler_type", "linear")
    if not isinstance(scheduler_type, str):
        raise ValueError("Invalid 'training.scheduler_type'")
    optimizer_type = training.get("optimizer_type", "adamw")
    if not isinstance(optimizer_type, str):
        raise ValueError("Invalid 'training.optimizer_type'")

    cfg: dict[str, Any] = {
        "dataset": dataset,
        "data_path": data_path,
        "split": _optional_str(payload, "split"),
        "attacker_extensions": _optional_str(payload, "attacker_extensions"),
        "model_name": model_name,
        "max_records": _optional_int(payload, "max_records"),
        "max_length": int(payload.get("max_length", 512)),
        "device": payload.get("device"),
        "output_root": output_root,
        "run_name": _optional_str(payload, "run_name"),
        "training": {
            "finetuning_epochs": int(training.get("finetuning_epochs", 15)),
            "batch_size": int(training.get("batch_size", 16)),
            "lr": float(lr),
            "use_pretraining": bool(training.get("use_pretraining", False)),
            "pretraining_epochs": int(training.get("pretraining_epochs", 3)),
            "early_stop_threshold": _optional_float(training, "early_stop_threshold"),
            "early_stop_patience": _optional_int_or_none(training, "early_stop_patience"),
            "loss_type": str(training.get("loss_type", "cross_entropy")),
            "focal_gamma": float(training.get("focal_gamma", 2.0)),
            "focal_alpha": _optional_float(training, "focal_alpha"),
            "focal_ignore_pt": _optional_float(training, "focal_ignore_pt"),
            "exclude_stopwords": bool(training.get("exclude_stopwords", False)),
            "weight_decay": float(training.get("weight_decay", 0.01)),
            "scheduler_type": scheduler_type,
            "optimizer_type": optimizer_type,
            "warmup_steps": int(training.get("warmup_steps", 10)),
            "warmup_ratio": _optional_float(training, "warmup_ratio"),
            "debug_tri": bool(training.get("debug_tri", False)),
            },
    }
    return cfg

def main() -> int:
    parser = argparse.ArgumentParser(description="Train and evaluate TRI model for re-identification")
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
    parser.add_argument("--task_id", type=int, default=None)

    parser.add_argument("--dataset", type=str, default="tab", choices=available_datasets)
    parser.add_argument("--data_path", type=str, default=None)
    parser.add_argument("--split", type=str, default=None)
    parser.add_argument("--model_name", type=str, default="distilbert-base-uncased")
    parser.add_argument("--max_records", type=int, default=None)
    parser.add_argument("--finetuning_epochs", type=int, default=15)
    parser.add_argument("--pretraining_epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--use_pretraining", action="store_true")
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--attacker_extensions", type=str, default=None)
    parser.add_argument("--early_stop_threshold", type=float, default=None)
    parser.add_argument("--eval_on_original", action="store_true", help="Also evaluate on original record text (in addition to deidentified/rewritten)")
    parser.add_argument("--loss_type", type=str, default="cross_entropy", choices=["cross_entropy", "focal"])
    parser.add_argument("--focal_gamma", type=float, default=2.0)
    parser.add_argument("--focal_alpha", type=float, default=None)
    parser.add_argument("--focal_ignore_pt", type=float, default=None)
    parser.add_argument("--exclude_stopwords", action="store_true")
    parser.add_argument("--p_agg", type=str, default="avg", choices=["avg", "max"], help="Method to aggregate token-level scores into record-level score")
    parser.add_argument("--debug_tri", action="store_true")

    args = parser.parse_args()
    task_id = resolve_task_id(args.task_id)

    if args.training_in is not None:
        cfg = _load_training_config(PROJECT_ROOT, Path(args.training_in))
        _apply_set_overrides(cfg, args.set_overrides)
        dataset = str(cfg["dataset"])
        data_path = Path(apply_task_template(str(cfg["data_path"]), task_id))
        model_name = str(cfg["model_name"])
        max_records = cfg.get("max_records")
        attacker_extensions = apply_task_template(cfg.get("attacker_extensions"), task_id)
        device = cfg.get("device")
        max_length = int(cfg.get("max_length", 512))
        training = cfg.get("training")
        if not isinstance(training, dict):
            raise SystemExit("Invalid training config")
        finetuning_epochs = int(training.get("finetuning_epochs", 15))
        batch_size = int(training.get("batch_size", 16))
        learning_rate = float(training.get("lr", 5e-5))
        use_pretraining = bool(training.get("use_pretraining", False))
        pretraining_epochs = int(training.get("pretraining_epochs", 3))
        early_stop_threshold = training.get("early_stop_threshold")
        early_stop_patience = training.get("early_stop_patience")
        loss_type = str(training.get("loss_type", "cross_entropy"))
        focal_gamma = float(training.get("focal_gamma", 2.0))
        focal_alpha = training.get("focal_alpha")
        focal_ignore_pt = training.get("focal_ignore_pt")
        exclude_stopwords = bool(training.get("exclude_stopwords", False))
        weight_decay = float(training.get("weight_decay", 0.01))
        scheduler_type = str(training.get("scheduler_type", "linear"))
        optimizer_type = str(training.get("optimizer_type", "adamw"))
        warmup_steps = int(training.get("warmup_steps", 10))
        warmup_ratio = training.get("warmup_ratio")
        p_agg = training.get("p_agg", "avg")
        debug_tri = bool(training.get("debug_tri", False))

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = apply_task_template(args.run_name, task_id) or apply_task_template(cfg.get("run_name"), task_id) or timestamp
        output_root_raw = apply_task_template(args.output_root, task_id) if args.output_root else apply_task_template(str(cfg["output_root"]), task_id)
        output_root = Path(output_root_raw).expanduser().resolve()
        split = apply_task_template(cfg.get("split"), task_id)
        if task_id is not None and f"__task_{task_id}" not in run_name:
            run_name = f"{run_name}__task_{task_id}"
        model_path: Path = (output_root / str(run_name)).resolve()
    else:
        if args.data_path is None:
            raise SystemExit("--data_path is required unless --training-in is provided")
        dataset = args.dataset
        data_path = Path(apply_task_template(args.data_path, task_id))
        model_name = args.model_name
        max_records = args.max_records
        attacker_extensions = apply_task_template(args.attacker_extensions, task_id)
        device = args.device
        max_length = 512
        finetuning_epochs = args.finetuning_epochs
        batch_size = args.batch_size
        learning_rate = args.learning_rate
        use_pretraining = bool(args.use_pretraining)
        pretraining_epochs = args.pretraining_epochs
        early_stop_threshold = args.early_stop_threshold
        early_stop_patience = None
        loss_type = args.loss_type
        focal_gamma = args.focal_gamma
        focal_alpha = args.focal_alpha
        focal_ignore_pt = args.focal_ignore_pt
        exclude_stopwords = bool(args.exclude_stopwords)
        weight_decay = 0.01
        scheduler_type = "linear"
        optimizer_type = "adamw"
        warmup_steps = 10
        warmup_ratio = None
        p_agg = args.p_agg
        debug_tri = bool(args.debug_tri)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_root_raw = apply_task_template(args.output_root, task_id) if args.output_root else f"models/tri_pipelines/{dataset}"
        output_root = Path(output_root_raw).expanduser().resolve()
        run_name = apply_task_template(args.run_name, task_id) or timestamp
        split = apply_task_template(args.split, task_id)
        if task_id is not None and f"__task_{task_id}" not in run_name:
            run_name = f"{run_name}__task_{task_id}"
        model_path = Path(apply_task_template(args.model_path, task_id)).expanduser().resolve() if args.model_path else (output_root / run_name).resolve()

    adapter_kwargs: dict[str, Any] = {
        "data": dataset,
        "data_in": str(data_path),
        "max_records": max_records,
    }
    if split is not None:
        adapter_kwargs["split"] = split
    adapter = get_attacker_adapter(dataset, **adapter_kwargs)
    if attacker_extensions:
        adapter.load_cache_from_jsonl(str(attacker_extensions))
    records = list(adapter.iter_records())
    if not records:
        raise SystemExit("No records loaded")

    tri = TRIDetector(dataset_name=dataset, model_name=model_name, max_length=max_length, device=device, p_agg=p_agg)

    if args.mode == "train":
        if args.model_path:
            base_path = Path(apply_task_template(args.model_path, task_id))
            if not base_path.exists():
                raise ValueError(f"Model path not found: {base_path}")
            tri.load(str(base_path))
        tri.setup(
            records=records,
            exclude_stopwords=exclude_stopwords,
        )
        model_path.mkdir(parents=True, exist_ok=True)
        tri.train(
            epochs=finetuning_epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            output_dir=str(model_path),
            use_pretraining=use_pretraining,
            pretraining_epochs=pretraining_epochs,
            early_stop_threshold=early_stop_threshold,
            early_stop_patience=early_stop_patience,
            loss_type=loss_type,
            focal_gamma=focal_gamma,
            focal_alpha=focal_alpha,
            focal_ignore_pt=focal_ignore_pt,
            weight_decay=weight_decay,
            scheduler_type=scheduler_type,
            optimizer_type=optimizer_type,
            warmup_steps=warmup_steps,
            warmup_ratio=warmup_ratio,
            debug_tri=debug_tri,
        )
        print(str(model_path))
        return 0

    if args.model_path is None:
        raise SystemExit("--model_path is required")
    tri.load(str(Path(apply_task_template(args.model_path, task_id)).expanduser().resolve()))
    tri.setup(records=records, exclude_stopwords=exclude_stopwords)

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
