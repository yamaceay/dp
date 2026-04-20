from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Optional, Tuple

from dp.loaders import get_adapter
from dp.utils.pii_detector import PIIDetector

import yaml


PROJECT_ROOT = Path(__file__).resolve().parent


def _parse_scalar(raw: str) -> Any:
    v = raw.strip()
    low = v.lower()
    if low in {"null", "none"}:
        return None
    if low == "true":
        return True
    if low == "false":
        return False
    try:
        if v.startswith("0") and len(v) > 1 and v[1].isdigit() and not v.startswith("0."):
            raise ValueError
        return int(v)
    except ValueError:
        pass
    try:
        return float(v)
    except ValueError:
        return v


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


def _read_yaml_mapping(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML root must be a mapping: {path}")
    return data


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


def _get_sequence(payload: dict[str, Any], key: str) -> tuple[str, ...]:
    value = payload.get(key)
    if value is None:
        return ()
    if not isinstance(value, list) or not value:
        raise ValueError(f"Invalid '{key}'")
    out: list[str] = []
    for item in value:
        if not isinstance(item, str) or not item.strip():
            raise ValueError(f"Invalid '{key}'")
        out.append(item)
    return tuple(out)


def _infer_output_root_from_env(env_path: Path, dataset: str) -> Optional[Path]:
    env = _read_yaml_mapping(env_path)
    block = env.get(dataset)
    if not isinstance(block, dict):
        return None
    root = block.get("pii_root")
    if isinstance(root, str) and root.strip():
        return Path(root).expanduser()
    annot = block.get("pii_annotator")
    if isinstance(annot, str) and annot.strip():
        return Path(annot).expanduser().parent
    return None


def _resolve_path(project_root: Path, raw: str) -> Path:
    p = Path(raw).expanduser()
    if not p.is_absolute():
        p = (project_root / p).resolve()
    return p


def load_splits(dataset: str, data_path: Path, max_records: Optional[int], data_ext: str) -> Tuple[list, list, list]:
    def load_one(name: str) -> list:
        path = data_path / f"{name}.{data_ext}"
        if not path.exists():
            return []
        adapter = get_adapter(dataset, data_in=str(path), max_records=max_records)
        return list(adapter.iter_records())

    train = load_one("train")
    dev = load_one("dev")
    test = load_one("test")
    return train, dev, test


def _normalize_grid_section(section: Any) -> dict[str, list[Any]]:
    if section is None:
        return {}
    if not isinstance(section, dict):
        raise ValueError("grid sections must be mappings")
    out: dict[str, list[Any]] = {}
    for key, value in section.items():
        name = str(key)
        if isinstance(value, list):
            if not value:
                raise ValueError(f"grid[{name}] cannot be empty")
            out[name] = value
        else:
            out[name] = [value]
    return out


def _product_grid(grid: dict[str, list[Any]]) -> Iterable[dict[str, Any]]:
    if not grid:
        yield {}
        return
    keys = list(grid.keys())
    values_lists = [grid[k] for k in keys]
    import itertools

    for values in itertools.product(*values_lists):
        yield {k: v for k, v in zip(keys, values)}


def _product_grid_grouped(grouped: dict[str, dict[str, list[Any]]]) -> Iterable[dict[str, dict[str, Any]]]:
    training = list(_product_grid(grouped.get("training", {})))
    evaluation = list(_product_grid(grouped.get("evaluation", {})))
    selection = list(_product_grid(grouped.get("selection", {})))
    for t in training:
        for e in evaluation:
            for s in selection:
                yield {"training": t, "evaluation": e, "selection": s}


def _pick_metric(metrics: dict[str, Any], select_metric: str) -> Optional[float]:
    candidates = [select_metric, f"eval_{select_metric}"]
    for key in candidates:
        val = metrics.get(key)
        if isinstance(val, (int, float)):
            return float(val)
    return None


def _better(a: Optional[float], b: Optional[float], direction: str) -> bool:
    if b is None:
        return a is not None
    if a is None:
        return False
    if direction == "min":
        return a < b
    return a > b


def _safe_run_name(params: dict[str, Any]) -> str:
    def fmt(v: Any) -> str:
        if isinstance(v, float):
            return f"{v:.8g}".replace("+", "")
        return str(v)

    parts = [f"{k}={fmt(params[k])}" for k in sorted(params.keys())]
    name = "__".join(parts) if parts else "default"
    return name.replace("/", "_").replace(" ", "_")


def _load_training_config(project_root: Path, config_path: Path, env_path: Optional[Path]) -> dict[str, Any]:
    resolved = config_path
    if not resolved.is_absolute():
        resolved = (project_root / resolved).resolve()
    payload = _read_yaml_mapping(resolved)

    dataset = payload.get("dataset", "tab")
    if not isinstance(dataset, str) or not dataset.strip():
        raise ValueError("Invalid 'dataset'")

    data_path = _resolve_path(project_root, _require_str(payload, "data_path"))
    model_name = _require_str(payload, "model_name")

    output_root: Optional[Path] = None
    output_root_raw = payload.get("output_root")
    if isinstance(output_root_raw, str) and output_root_raw.strip():
        output_root = _resolve_path(project_root, output_root_raw)
    if output_root is None and env_path is not None:
        output_root = _infer_output_root_from_env(env_path, dataset)
    if output_root is None:
        output_root = (project_root / "models" / "pii_detectors" / dataset).resolve()

    training_raw = _get_mapping(payload, "training")
    evaluation_raw = _get_mapping(payload, "evaluation")
    selection_raw = _get_mapping(payload, "selection")

    modes = _get_sequence(evaluation_raw, "modes")
    if not modes:
        modes = ("strict", "exact", "partial", "type")

    cfg: dict[str, Any] = {
        "dataset": dataset,
        "data_path": data_path,
        "model_name": model_name,
        "output_root": output_root,
        "run_name": _optional_str(payload, "run_name"),
        "max_records": _optional_int(payload, "max_records"),
        "max_length": int(payload.get("max_length", 512)),
        "use_chunking": bool(payload.get("use_chunking", True)),
        "training": {
            "epochs": int(training_raw.get("epochs", 5)),
            "batch_size": int(training_raw.get("batch_size", 8)),
            "learning_rate": float(training_raw.get("learning_rate", 2e-5)),
            "weight_decay": float(training_raw.get("weight_decay", 0.01)),
            "warmup_steps": int(training_raw.get("warmup_steps", 100)),
            "logging_steps": int(training_raw.get("logging_steps", 50)),
            "eval_steps": int(training_raw.get("eval_steps", 250)),
            "save_steps": int(training_raw.get("save_steps", 250)),
            "save_total_limit": _optional_int(training_raw, "save_total_limit"),
            "gradient_accumulation_steps": int(training_raw.get("gradient_accumulation_steps", 1)),
            "seed": _optional_int(training_raw, "seed"),
            "fp16": _optional_bool(training_raw, "fp16"),
        },
        "evaluation": {
            "use_nervaluate": bool(evaluation_raw.get("use_nervaluate", True)),
            "modes": modes,
            "primary_mode": str(evaluation_raw.get("primary_mode", "partial")),
        },
        "selection": {
            "metric": str(selection_raw.get("metric", "partial_f1")),
            "direction": str(selection_raw.get("direction", "max")),
        },
    }
    return cfg


def _load_grid_config(project_root: Path, grid_path: Path) -> dict[str, dict[str, list[Any]]]:
    resolved = grid_path
    if not resolved.is_absolute():
        resolved = (project_root / resolved).resolve()
    payload = _read_yaml_mapping(resolved)
    grid = payload.get("grid")
    if not isinstance(grid, dict):
        raise ValueError("Missing or invalid 'grid'")
    training = _normalize_grid_section(grid.get("training"))
    evaluation = _normalize_grid_section(grid.get("evaluation"))
    selection = _normalize_grid_section(grid.get("selection"))
    flat = {k: v for k, v in grid.items() if k not in {"training", "evaluation", "selection"}}
    if flat:
        raise ValueError("grid must group keys under training/evaluation/selection")
    return {"training": training, "evaluation": evaluation, "selection": selection}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="train", choices=["train", "evaluate", "predict", "grid"])
    parser.add_argument("--training-in", type=str, default=None)
    parser.add_argument("--grid-in", type=str, default=None)
    parser.add_argument("--env-in", type=str, default=None)
    parser.add_argument("--output-root", type=str, default=None)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--max-runs", type=int, default=None)
    parser.add_argument(
        "--set",
        dest="set_overrides",
        action="append",
        default=None,
        help="Override config values, e.g. --set training.learning_rate=1e-5 --set selection.metric=partial_recall",
    )
    args = parser.parse_args()

    env_path = Path(args.env_in).resolve() if args.env_in else None

    if args.mode == "grid":
        raise SystemExit(
            "grid mode is deprecated for Slurm sweeps. "
            "Use --mode train with repeated --set overrides and let the Slurm table enumerate combinations."
        )

    if args.mode == "train":
        if args.training_in is None:
            raise SystemExit("--training-in is required for train")

        cfg = _load_training_config(PROJECT_ROOT, Path(args.training_in), env_path)
        _apply_set_overrides(cfg, args.set_overrides)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = args.run_name or cfg.get("run_name") or timestamp
        output_root = Path(args.output_root).expanduser().resolve() if args.output_root else cfg["output_root"]
        output_dir = (output_root / run_name).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)


        train_records, dev_records, test_records = load_splits(
            str(cfg["dataset"]), Path(cfg["data_path"]), cfg.get("max_records"), cfg.get("data_ext")
        )
        if not train_records:
            raise SystemExit("Missing train split")

        detector = PIIDetector(
            model_name=str(cfg["model_name"]),
            max_length=int(cfg["max_length"]),
            use_chunking=bool(cfg["use_chunking"]),
        )
        detector.set_train_dataset(train_records)
        if dev_records:
            detector.set_val_dataset(dev_records)
        if test_records:
            detector.set_test_dataset(test_records)

        detector.train(
            epochs=int(cfg["training"]["epochs"]),
            batch_size=int(cfg["training"]["batch_size"]),
            learning_rate=float(cfg["training"]["learning_rate"]),
            weight_decay=float(cfg["training"]["weight_decay"]),
            warmup_steps=int(cfg["training"]["warmup_steps"]),
            logging_steps=int(cfg["training"]["logging_steps"]),
            eval_steps=int(cfg["training"]["eval_steps"]),
            save_steps=int(cfg["training"]["save_steps"]),
            save_total_limit=cfg["training"]["save_total_limit"],
            gradient_accumulation_steps=int(cfg["training"]["gradient_accumulation_steps"]),
            seed=cfg["training"]["seed"],
            fp16=cfg["training"]["fp16"],
            output_dir=str(output_dir),
            use_nervaluate=bool(cfg["evaluation"]["use_nervaluate"]),
            nervaluate_modes=list(cfg["evaluation"]["modes"]),
            primary_nervaluate_mode=str(cfg["evaluation"]["primary_mode"]),
            metric_for_best_model=str(cfg["selection"]["metric"]),
        )
        print(str(output_dir))
        return 0

    if args.model_path is None:
        raise SystemExit("--model-path is required")

    detector = PIIDetector(model_name=args.model_path)

    if args.training_in is None:
        raise SystemExit("--training-in is required")

    cfg = _load_training_config(PROJECT_ROOT, Path(args.training_in), env_path)
    _apply_set_overrides(cfg, args.set_overrides)
    _, _, test_records = load_splits(str(cfg["dataset"]), Path(cfg["data_path"]), cfg.get("max_records"), cfg.get("data_ext"))
    if not test_records:
        raise SystemExit("Missing test split")

    if args.mode == "evaluate":
        metrics = detector.evaluate(
            test_records,
            use_nervaluate=bool(cfg["evaluation"]["use_nervaluate"]),
            modes=list(cfg["evaluation"]["modes"]),
        )
        print(metrics)
        return 0

    sample = test_records[:5]
    predicted = detector.predict(sample)
    print([{"uid": r.uid, "spans": len(r.spans or [])} for r in predicted])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
