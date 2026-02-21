from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from dp.experiments import ExperimentResult
from dp.experiments.utils import OutputCallback


@dataclass(frozen=True)
class UtilityEvaluationReport:
    name: str
    source: Optional[Path]
    valid: bool
    metrics: Dict[str, float]
    drops: Dict[str, float]
    train_matched: int
    train_total: int
    test_matched: int
    test_total: int
    available: int
    train_results: Dict[str, Any]
    val_results: Dict[str, Any]
    test_results: Dict[str, Any]
    overall_results: Dict[str, Any]
    grouped_results: Dict[str, Any]


@dataclass(frozen=True)
class UtilityExperimentReport:
    model_name: str
    primary_metric: str
    baseline_metrics: Dict[str, float]
    baseline_train_size: int
    baseline_test_size: int
    baseline_train_metrics: Dict[str, float]
    baseline_test_metrics: Dict[str, float]
    baseline_overall_metrics: Dict[str, float]
    baseline_median_dummy_mae: Dict[str, Any]
    evaluations: List[UtilityEvaluationReport]


class UtilityReportOutputter:
    def __init__(self, sink: OutputCallback):
        self.sink = sink

    def output(self, report: UtilityExperimentReport) -> None:
        raise NotImplementedError


class JsonLinesUtilityReportOutputter(UtilityReportOutputter):
    def output(self, report: UtilityExperimentReport) -> None:
        records: List[Dict[str, Any]] = [
            {
                "type": "experiment",
                "model": report.model_name,
                "primary_metric": report.primary_metric,
                "baseline_metrics": report.baseline_metrics,
                "baseline_train_size": report.baseline_train_size,
                "baseline_test_size": report.baseline_test_size,
                "baseline_train_metrics": report.baseline_train_metrics,
                "baseline_test_metrics": report.baseline_test_metrics,
                "baseline_overall_metrics": report.baseline_overall_metrics,
                "baseline_median_dummy_mae": report.baseline_median_dummy_mae,
            }
        ]
        for evaluation in report.evaluations:
            records.append(
                {
                    "type": "evaluation",
                    "name": evaluation.name,
                    "source": str(evaluation.source) if evaluation.source else None,
                    "valid": evaluation.valid,
                    "metrics": evaluation.metrics,
                    "drops": evaluation.drops,
                    "train_matched": evaluation.train_matched,
                    "train_total": evaluation.train_total,
                    "test_matched": evaluation.test_matched,
                    "test_total": evaluation.test_total,
                    "available": evaluation.available,
                    "train_results": evaluation.train_results,
                    "val_results": evaluation.val_results,
                    "test_results": evaluation.test_results,
                    "overall_results": evaluation.overall_results,
                    "grouped_results": evaluation.grouped_results,
                }
            )
        serialized = "\n".join(json.dumps(record, ensure_ascii=False) for record in records)
        self.sink(serialized)


def build_utility_report(result: ExperimentResult, sources: Dict[str, Path]) -> UtilityExperimentReport:
    metrics = result.metrics or {}
    model_name = str(metrics.get("model", ""))
    primary_metric = str(metrics.get("primary_metric", ""))
    baseline_payload = metrics.get("baseline", {}) or {}
    baseline_metrics_raw = baseline_payload.get("metrics", {}) or {}
    baseline_metrics = {name: float(value) for name, value in baseline_metrics_raw.items()}
    baseline_train = int(baseline_payload.get("train_size", 0))
    baseline_test = int(baseline_payload.get("test_size", 0))
    baseline_train_metrics = {key: float(value) for key, value in (baseline_payload.get("train_metrics", {}) or {}).items()}
    baseline_test_metrics = {key: float(value) for key, value in (baseline_payload.get("test_metrics", {}) or {}).items()}
    baseline_overall_metrics = {key: float(value) for key, value in (baseline_payload.get("overall_metrics", {}) or {}).items()}
    baseline_median_dummy_mae = baseline_payload.get("median_dummy_mae", {}) or {}
    if not isinstance(baseline_median_dummy_mae, dict):
        raise ValueError("baseline.median_dummy_mae must be a mapping")
    evaluation_metrics: Dict[str, Dict[str, Any]] = metrics.get("evaluations", {})
    evaluations: List[UtilityEvaluationReport] = []
    for name in sorted(evaluation_metrics.keys()):
        payload = evaluation_metrics.get(name) or {}
        metrics_payload = payload.get("metrics", {}) or {}
        drops_payload = payload.get("drops", {}) or {}
        train_results = payload.get("train_results", {}) or {}
        val_results = payload.get("val_results", {}) or {}
        test_results = payload.get("test_results", {}) or {}
        overall_results = payload.get("overall_results", {}) or {}
        grouped_results = payload.get("grouped_results", {}) or {}
        evaluations.append(
            UtilityEvaluationReport(
                name=name,
                source=sources.get(name),
                valid=bool(payload.get("valid")),
                metrics={key: float(value) for key, value in metrics_payload.items()},
                drops={key: float(value) for key, value in drops_payload.items()},
                train_matched=int(payload.get("train_matched", 0)),
                train_total=int(payload.get("train_total", 0)),
                test_matched=int(payload.get("test_matched", 0)),
                test_total=int(payload.get("test_total", 0)),
                available=int(payload.get("available", 0)),
                train_results=train_results,
                val_results=val_results,
                test_results=test_results,
                overall_results=overall_results,
                grouped_results=grouped_results,
            )
        )
    return UtilityExperimentReport(
        model_name=model_name,
        primary_metric=primary_metric,
        baseline_metrics=baseline_metrics,
        baseline_train_size=baseline_train,
        baseline_test_size=baseline_test,
        baseline_train_metrics=baseline_train_metrics,
        baseline_test_metrics=baseline_test_metrics,
        baseline_overall_metrics=baseline_overall_metrics,
        baseline_median_dummy_mae=baseline_median_dummy_mae,
        evaluations=evaluations,
    )


def create_utility_outputter(fmt: str, sink: OutputCallback) -> UtilityReportOutputter:
    if fmt == "jsonl":
        return JsonLinesUtilityReportOutputter(sink)
    raise ValueError(f"Unsupported output format '{fmt}'")
