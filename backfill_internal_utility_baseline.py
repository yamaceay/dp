#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List

from dp.experiments.config import align_evaluation_texts, load_config, select_records
from dp.experiments.handlers import (
    _prepare_utility,
    _render_component_paths,
    _resolve_utility_components,
    load_records,
    merge_params,
    normalize_output_settings,
)
from dp.experiments.utility.internal_utility import (
    InternalUtilityConfig,
    _score_difference,
    compute_internal_utility_baselines,
)
from dp.experiments.utils import collect_jsonl_sources


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if not text:
            continue
        rows.append(json.loads(text))
    return rows


def _write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    payload = "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n"
    path.write_text(payload, encoding="utf-8")


def _build_internal_cfg(ctx: Any, params: Dict[str, Any]) -> InternalUtilityConfig:
    internal_cfg_raw = params.get("internal_utility", {}) or {}
    if not isinstance(internal_cfg_raw, dict):
        raise ValueError("internal_utility config must be a mapping")
    return InternalUtilityConfig(
        n_folds=int(internal_cfg_raw.get("n_folds", 50)),
        eval_fold_offset=int(internal_cfg_raw.get("eval_fold_offset", 1)),
        random_state=int(internal_cfg_raw.get("random_state", ctx.random_state)),
        max_rounds=int(internal_cfg_raw["max_rounds"]) if internal_cfg_raw.get("max_rounds") is not None else None,
    )


def _patch_record_with_baseline(record: Dict[str, Any], baseline_summary: Dict[str, Any]) -> None:
    metrics = record.get("metrics")
    if isinstance(metrics, dict):
        record["drops"] = _score_difference(
            baseline_summary.get("metrics", {}) or {},
            {k: float(v) for k, v in metrics.items()},
        )
    for key in ("train_results", "test_results", "overall_results"):
        payload = record.get(key)
        if not isinstance(payload, dict):
            continue
        current_metrics = payload.get("metrics", {}) or {}
        baseline_metrics = ((baseline_summary.get(key, {}) or {}).get("metrics", {}) or {})
        payload["baseline_metrics"] = {k: float(v) for k, v in baseline_metrics.items()}
        payload["drops"] = _score_difference(
            {k: float(v) for k, v in baseline_metrics.items()},
            {k: float(v) for k, v in current_metrics.items()} if isinstance(current_metrics, dict) else {},
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="Backfill internal utility original-text baselines into JSONL logs")
    parser.add_argument("--config", required=True, type=str)
    parser.add_argument("--log", required=True, type=str)
    parser.add_argument("--task-id", type=int, default=None)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    cfg_for_merge = dict(config)
    if args.task_id is not None:
        cfg_for_merge["task_id"] = int(args.task_id)
        os.environ["TASK_ID"] = str(args.task_id)
        os.environ["TASK_WITHIN_JOB"] = str(args.task_id)

    params = merge_params(cfg_for_merge, SimpleNamespace(identifier=None))
    normalize_output_settings(params)
    if str(params.get("protocol", "")).strip().lower() != "internal_utility":
        raise SystemExit("config must be an internal_utility utility config")
    ctx = _prepare_utility(params)
    if not ctx.annotations:
        raise SystemExit("annotations are required")

    records = load_records(ctx.dataset, ctx.data_in, params.get("max_records"), split=ctx.split)
    records = select_records(records, ctx.selection_criteria)
    if not records:
        raise SystemExit("no records selected")

    sources = collect_jsonl_sources(*ctx.annotations)
    if not sources:
        raise SystemExit("no anonymized output files discovered")
    evaluation_texts = align_evaluation_texts(records, sources)
    if not evaluation_texts:
        raise SystemExit("no anonymized texts aligned with dataset records")

    vec_name, vec_kwargs, head_name, head_kwargs = _resolve_utility_components(ctx.spec, params)
    vec_kwargs = _render_component_paths(vec_kwargs, ctx.task_id)
    head_kwargs = _render_component_paths(head_kwargs, ctx.task_id)
    internal_cfg = _build_internal_cfg(ctx, params)

    baselines = compute_internal_utility_baselines(
        spec=ctx.spec,
        records=records,
        evaluation_texts=evaluation_texts,
        vectorizer_name=vec_name or None,
        vectorizer_kwargs=vec_kwargs,
        head_name=head_name or None,
        head_kwargs=head_kwargs,
        identifier=ctx.identifier,
        config=internal_cfg,
    )

    log_path = Path(args.log)
    rows = _read_jsonl(log_path)
    if not rows:
        raise SystemExit(f"empty log file: {log_path}")

    global_baseline = baselines["global"]
    per_eval = baselines["per_evaluation"]

    for row in rows:
        row_type = row.get("type")
        if row_type == "experiment":
            row["baseline_metrics"] = dict(global_baseline.get("metrics", {}) or {})
            row["baseline_train_size"] = int(global_baseline.get("train_matched", 0))
            row["baseline_test_size"] = int(global_baseline.get("test_matched", 0))
            row["baseline_train_metrics"] = dict((global_baseline.get("train_results", {}) or {}).get("metrics", {}) or {})
            row["baseline_test_metrics"] = dict((global_baseline.get("test_results", {}) or {}).get("metrics", {}) or {})
            row["baseline_overall_metrics"] = dict((global_baseline.get("overall_results", {}) or {}).get("metrics", {}) or {})
            if "baseline_median_dummy_mae" not in row or not isinstance(row["baseline_median_dummy_mae"], dict):
                row["baseline_median_dummy_mae"] = {}
            continue
        if row_type != "evaluation":
            continue
        name = row.get("name")
        if not isinstance(name, str):
            continue
        baseline_summary = per_eval.get(name)
        if not isinstance(baseline_summary, dict):
            continue
        _patch_record_with_baseline(row, baseline_summary)

    out_path = Path(args.output) if args.output else log_path
    _write_jsonl(out_path, rows)
    print(f"Patched baseline into {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
