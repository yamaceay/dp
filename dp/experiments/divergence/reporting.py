from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from dp.experiments import ExperimentResult
from dp.experiments.utils import OutputCallback


@dataclass(frozen=True)
class DivergenceEntry:
    key: str
    index: int
    name: Optional[str]
    similarity: float
    divergence: float


@dataclass(frozen=True)
class DivergenceEvaluationReport:
    name: str
    source: Optional[Path]
    matched_count: int
    available_count: int
    entries: List[DivergenceEntry]
    summary: Optional[Dict[str, float]]


@dataclass(frozen=True)
class DivergenceExperimentReport:
    score: float
    metric_name: str
    metric_metadata: Dict[str, Any]
    original_record_count: int
    evaluations: List[DivergenceEvaluationReport]


class DivergenceReportOutputter:
    def __init__(self, sink: OutputCallback):
        self.sink = sink

    def output(self, report: DivergenceExperimentReport) -> None:
        raise NotImplementedError


class JsonLinesDivergenceReportOutputter(DivergenceReportOutputter):
    def output(self, report: DivergenceExperimentReport) -> None:
        records: List[Dict[str, Any]] = [
            {
                "type": "experiment",
                "score": report.score,
                "metric_name": report.metric_name,
                "metric_metadata": report.metric_metadata,
                "original_record_count": report.original_record_count,
            }
        ]
        for evaluation in report.evaluations:
            records.append(
                {
                    "type": "evaluation",
                    "name": evaluation.name,
                    "matched_count": evaluation.matched_count,
                    "available_count": evaluation.available_count,
                    "source": str(evaluation.source) if evaluation.source else None,
                    "summary": evaluation.summary,
                }
            )
            for entry in evaluation.entries:
                records.append(
                    {
                        "type": "entry",
                        "evaluation": evaluation.name,
                        "key": entry.key,
                        "index": entry.index,
                        "name": entry.name,
                        "similarity": entry.similarity,
                        "divergence": entry.divergence,
                    }
                )
        serialized = "\n".join(json.dumps(record, ensure_ascii=False) for record in records)
        self.sink(serialized)


def build_divergence_report(
    result: ExperimentResult,
    original_record_count: int,
    evaluation_sources: Dict[str, Path],
) -> DivergenceExperimentReport:
    metrics = result.metrics or {}
    metric_name = str(metrics.get("metric", ""))
    metric_metadata = metrics.get("metric_metadata", {}) or {}
    record_info: Dict[str, Dict[str, Any]] = metrics.get("records", {})
    evaluation_metrics: Dict[str, Dict[str, Any]] = metrics.get("evaluations", {})
    evaluations: List[DivergenceEvaluationReport] = []
    for name in sorted(evaluation_metrics.keys()):
        payload = evaluation_metrics[name] or {}
        summary = payload.get("summary")
        matched = int(payload.get("matched", 0))
        available = int(payload.get("total", matched))
        similarities: Dict[str, float] = payload.get("similarity", {})
        divergences: Dict[str, float] = payload.get("divergence", {})

        def sort_key(item: str) -> int:
            info = record_info.get(item, {})
            return int(info.get("index", 0))

        entries: List[DivergenceEntry] = []
        for key in sorted(similarities.keys(), key=sort_key):
            info = record_info.get(key, {})
            entries.append(
                DivergenceEntry(
                    key=key,
                    index=int(info.get("index", 0)),
                    name=info.get("name"),
                    similarity=float(similarities[key]),
                    divergence=float(divergences.get(key, 0.0)),
                )
            )
        evaluations.append(
            DivergenceEvaluationReport(
                name=name,
                source=evaluation_sources.get(name),
                matched_count=matched,
                available_count=available,
                entries=entries,
                summary=summary,
            )
        )
    return DivergenceExperimentReport(
        score=float(result.score),
        metric_name=metric_name,
        metric_metadata=metric_metadata,
        original_record_count=original_record_count,
        evaluations=evaluations,
    )


def create_divergence_outputter(fmt: str, sink: OutputCallback) -> DivergenceReportOutputter:
    if fmt == "jsonl":
        return JsonLinesDivergenceReportOutputter(sink)
    raise ValueError(f"Unsupported output format '{fmt}'")
