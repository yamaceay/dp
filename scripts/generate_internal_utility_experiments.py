#!/usr/bin/env python3

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

import yaml


ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = ROOT / "configs" / "5_experiments"
TABLE_ROOT = ROOT / "slurm" / "tables"
DATASETS = ("db_bio", "tab", "reddit")


def _read_yaml(path: Path) -> Dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML root must be a mapping: {path}")
    return data


def _write_yaml(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = yaml.safe_dump(data, sort_keys=False, allow_unicode=False)
    path.write_text(text, encoding="utf-8")


def _internal_output_file(dataset: str, label: str) -> str:
    if dataset in {"db_bio", "tab"}:
        return f"logs/internal_utility/{dataset}_{label}_task_{{task_id}}.jsonl"
    return f"logs/internal_utility/{dataset}_{label}.jsonl"


def _build_internal_config(dataset: str, label: str, base: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    out["protocol"] = "internal_utility"
    out["internal_utility"] = {
        "n_folds": 50,
    }
    if dataset in {"db_bio", "tab"}:
        out["internal_utility"]["random_state"] = "{task_id}"
    out.pop("source_splits", None)
    if dataset not in {"db_bio", "tab"}:
        out.pop("split", None)
    target = out.get("target") or {}
    target_type = "nominal"
    if isinstance(target, dict):
        target_type = str(target.get("type", "nominal")).strip().lower() or "nominal"
    elif isinstance(target, str):
        target_type = "nominal"
    out["vectorizer"] = {
        "type": "tfidf",
        "params": {
            "ngram_range": [1, 2],
            "max_features": 10000,
            "min_df": 2,
        },
    }
    if target_type == "cardinal":
        out["head"] = {
            "type": "linear_regressor",
            "params": {
                "primary_metric": "r2",
            },
        }
    else:
        out["head"] = {
            "type": "logistic_classifier",
            "params": {
                "max_iter": 500,
                "class_weight": "balanced",
                "n_jobs": -1,
                "primary_metric": "f1",
            },
        }
    output = dict(out.get("output") or {})
    output["format"] = "jsonl"
    output["file"] = _internal_output_file(dataset, label)
    out["output"] = output
    if dataset in {"db_bio", "tab"}:
        out["task_id"] = "{task_id}"
    return out


def _collect_utility_configs(dataset: str) -> List[Path]:
    utility_dir = CONFIG_ROOT / dataset / "utility"
    if not utility_dir.exists():
        return []
    return sorted(p for p in utility_dir.glob("*.yaml") if not p.name.startswith("internal_"))


def _table_line(name: str, cmd: str, width: int) -> str:
    return f"{name.ljust(width)}|{cmd}"


def generate() -> Tuple[List[Path], List[Path]]:
    written_configs: List[Path] = []
    written_tables: List[Path] = []
    for dataset in DATASETS:
        cfg_paths = _collect_utility_configs(dataset)
        rows: List[Tuple[str, str]] = []
        for cfg_path in cfg_paths:
            label = cfg_path.stem
            internal_path = cfg_path.with_name(f"internal_{label}.yaml")
            base_cfg = _read_yaml(cfg_path)
            internal_cfg = _build_internal_config(dataset, label, base_cfg)
            _write_yaml(internal_path, internal_cfg)
            written_configs.append(internal_path)
            job_name = f"utility_internal_{dataset}_{label}"
            cmd = f"python run.py utility --config {internal_path.relative_to(ROOT).as_posix()}"
            rows.append((job_name, cmd))
        if not rows:
            continue
        width = max(len(name) for name, _ in rows)
        table_text = "\n".join(_table_line(name, cmd, width) for name, cmd in rows) + "\n"
        table_path = TABLE_ROOT / f"5x_{dataset}_experiments.table"
        table_path.parent.mkdir(parents=True, exist_ok=True)
        table_path.write_text(table_text, encoding="utf-8")
        written_tables.append(table_path)
    return written_configs, written_tables


def main() -> int:
    written_configs, written_tables = generate()
    print(f"configs={len(written_configs)}")
    for path in written_configs:
        print(path.relative_to(ROOT).as_posix())
    print(f"tables={len(written_tables)}")
    for path in written_tables:
        print(path.relative_to(ROOT).as_posix())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
