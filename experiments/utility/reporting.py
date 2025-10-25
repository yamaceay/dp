from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from experiments import ExperimentResult
from experiments.utils import OutputCallback


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


@dataclass(frozen=True)
class UtilityExperimentReport:
    score: float
    model_name: str
    primary_metric: str
    baseline_metrics: Dict[str, float]
    baseline_train_size: int
    baseline_test_size: int
    evaluations: List[UtilityEvaluationReport]


class UtilityReportOutputter:
    def __init__(self, sink: OutputCallback):
        self.sink = sink

    def output(self, report: UtilityExperimentReport) -> None:
        raise NotImplementedError


class TextUtilityReportOutputter(UtilityReportOutputter):
    def output(self, report: UtilityExperimentReport) -> None:
        lines: List[str] = [
            f"Score ({report.primary_metric}): {report.score:.4f}",
            f"Model: {report.model_name}",
            "",
            "Baseline",
        ]
        if report.baseline_metrics:
            baseline_text = " ".join(
                f"{name}={value:.4f}" for name, value in sorted(report.baseline_metrics.items())
            )
            lines.append(f"  {baseline_text}")
        else:
            lines.append("  none")
        lines.append(f"  train={report.baseline_train_size} test={report.baseline_test_size}")
        lines.append("")
        lines.append("Evaluation datasets")
        if not report.evaluations:
            lines.append("  none")
        else:
            for evaluation in report.evaluations:
                prefix = f"  {evaluation.name}: "
                if evaluation.valid and evaluation.metrics:
                    metrics_text = " ".join(
                        f"{name}={value:.4f}" for name, value in sorted(evaluation.metrics.items())
                    )
                    prefix += metrics_text
                    if evaluation.drops:
                        drops_text = " ".join(
                            f"{name}={value:.4f}" for name, value in sorted(evaluation.drops.items())
                        )
                        prefix += f" drops[{drops_text}]"
                else:
                    prefix += "insufficient coverage"
                prefix += (
                    f" (train {evaluation.train_matched}/{evaluation.train_total},"
                    f" test {evaluation.test_matched}/{evaluation.test_total})"
                )
                if evaluation.source:
                    prefix += f" from {evaluation.source}"
                lines.append(prefix)
        self.sink("\n".join(lines))


class JsonUtilityReportOutputter(UtilityReportOutputter):
    def output(self, report: UtilityExperimentReport) -> None:
        payload: Dict[str, Any] = {
            "score": report.score,
            "model": report.model_name,
            "primary_metric": report.primary_metric,
            "baseline": {
                "metrics": report.baseline_metrics,
                "train_size": report.baseline_train_size,
                "test_size": report.baseline_test_size,
            },
            "evaluations": [
                {
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
                }
                for evaluation in report.evaluations
            ],
        }
        self.sink(json.dumps(payload, ensure_ascii=False, indent=2))


class JsonLinesUtilityReportOutputter(UtilityReportOutputter):
    def output(self, report: UtilityExperimentReport) -> None:
        records: List[Dict[str, Any]] = [
            {
                "type": "experiment",
                "score": report.score,
                "model": report.model_name,
                "primary_metric": report.primary_metric,
                "baseline_metrics": report.baseline_metrics,
                "baseline_train_size": report.baseline_train_size,
                "baseline_test_size": report.baseline_test_size,
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
    evaluation_metrics: Dict[str, Dict[str, Any]] = metrics.get("evaluations", {})
    evaluations: List[UtilityEvaluationReport] = []
    for name in sorted(evaluation_metrics.keys()):
        payload = evaluation_metrics.get(name) or {}
        metrics_payload = payload.get("metrics", {}) or {}
        drops_payload = payload.get("drops", {}) or {}
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
            )
        )
    return UtilityExperimentReport(
        score=float(result.score),
        model_name=model_name,
        primary_metric=primary_metric,
        baseline_metrics=baseline_metrics,
        baseline_train_size=baseline_train,
        baseline_test_size=baseline_test,
        evaluations=evaluations,
    )


def create_utility_outputter(fmt: str, sink: OutputCallback) -> UtilityReportOutputter:
    if fmt == "text":
        return TextUtilityReportOutputter(sink)
    if fmt == "json":
        return JsonUtilityReportOutputter(sink)
    if fmt == "jsonl":
        return JsonLinesUtilityReportOutputter(sink)
    raise ValueError(f"Unsupported output format '{fmt}'")
