from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from dp.experiments import ExperimentResult
from dp.experiments.utils import OutputCallback


@dataclass(frozen=True)
class RankEntry:
    key: str
    rank: int
    name: Optional[str]
    index: int
    delta: Optional[int] = None


@dataclass(frozen=True)
class EvaluationReport:
    name: str
    source: Optional[Path]
    record_count: int
    ranks: List[RankEntry]
    summary: Optional[Dict[str, Any]]


@dataclass(frozen=True)
class PrivacyExperimentReport:
    score: Dict[str, float]
    original_record_count: int
    original_ranks: List[RankEntry]
    evaluations: List[EvaluationReport]


class PrivacyReportOutputter:
    def __init__(self, sink: OutputCallback):
        self.sink = sink

    def output(self, report: PrivacyExperimentReport) -> None:
        raise NotImplementedError


class JsonLinesPrivacyReportOutputter(PrivacyReportOutputter):
    def output(self, report: PrivacyExperimentReport) -> None:
        records: List[Dict[str, Any]] = [
            {
                "type": "experiment",
                "score": {
                    "mean": 0.0, 
                    "mean_reciprocal_rank": sum(1 / r.rank for r in report.original_ranks) / len(report.original_ranks),
                    "accuracy": sum(r.rank == 1 for r in report.original_ranks) / len(report.original_ranks),
                },
                "original_record_count": report.original_record_count,
            }
        ]
        for entry in report.original_ranks:
            records.append(self._rank_record("original_rank", entry))
        for evaluation in report.evaluations:
            privacy_metrics = {
                "mean_reciprocal_rank": sum(1 / r.rank for r in evaluation.ranks) / len(evaluation.ranks),
                "accuracy": sum(r.rank == 1 for r in evaluation.ranks) / len(evaluation.ranks),
            }
            records.append(
                {
                    "type": "evaluation",
                    "name": evaluation.name,
                    "record_count": evaluation.record_count,
                    "source": str(evaluation.source) if evaluation.source else None,
                    "summary": {
                        **(evaluation.summary or {}),
                        **privacy_metrics,
                    }
                }
            )
            for entry in evaluation.ranks:
                records.append(self._rank_record("evaluation_rank", entry, evaluation.name))
        serialized = "\n".join(json.dumps(record, ensure_ascii=False) for record in records)
        self.sink(serialized)

    @staticmethod
    def _rank_record(record_type: str, entry: RankEntry, evaluation: Optional[str] = None) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "type": record_type,
            "key": entry.key,
            "rank": entry.rank,
            "index": entry.index,
        }
        if entry.name:
            payload["name"] = entry.name
        if entry.delta is not None:
            payload["delta"] = entry.delta
        if evaluation:
            payload["evaluation"] = evaluation
        return payload


def build_privacy_report(
    result: ExperimentResult,
    original_record_count: int,
    annotation_sources: Dict[str, Path],
    evaluation_record_counts: Dict[str, int],
) -> PrivacyExperimentReport:
    metrics = result.metrics or {}
    record_info: Dict[str, Dict[str, Any]] = metrics.get("records", {})
    original_ranks_map: Dict[str, int] = metrics.get("original_ranks", {})
    original_ranks: List[RankEntry] = []
    for key, rank in sorted(
        original_ranks_map.items(),
        key=lambda item: record_info.get(item[0], {}).get("index", 0),
    ):
        info = record_info.get(key, {})
        original_ranks.append(
            RankEntry(
                key=key,
                rank=rank,
                name=info.get("name"),
                index=int(info.get("index", 0)),
            )
        )
    evaluations_metric: Dict[str, Dict[str, Any]] = metrics.get("evaluations", {})
    evaluation_reports: List[EvaluationReport] = []
    for name in sorted(evaluations_metric.keys()):
        payload = evaluations_metric[name] or {}
        ranks_map: Dict[str, int] = payload.get("ranks", {})
        deltas: Dict[str, int] = payload.get("rank_deltas", {})
        ranks: List[RankEntry] = []
        for key, rank in sorted(
            ranks_map.items(),
            key=lambda item: record_info.get(item[0], {}).get("index", 0),
        ):
            info = record_info.get(key, {})
            ranks.append(
                RankEntry(
                    key=key,
                    rank=rank,
                    name=info.get("name"),
                    index=int(info.get("index", 0)),
                    delta=deltas.get(key),
                )
            )
        summary = payload.get("rank_delta_summary") or {}
        if ranks:
            summary = {
                **summary,
                "mean_reciprocal_rank": sum(1 / r.rank for r in ranks) / len(ranks),
                "accuracy": sum(r.rank == 1 for r in ranks) / len(ranks),
            }
        evaluation_reports.append(
            EvaluationReport(
                name=name,
                source=annotation_sources.get(name),
                record_count=evaluation_record_counts.get(name, 0),
                ranks=ranks,
                summary=summary,
            )
        )
    return PrivacyExperimentReport(
        score={
            "mean": 0.0,
            "mean_reciprocal_rank": sum(1 / r.rank for r in original_ranks) / len(original_ranks),
            "accuracy": sum(r.rank == 1 for r in original_ranks) / len(original_ranks),
        },
        original_record_count=original_record_count,
        original_ranks=original_ranks,
        evaluations=evaluation_reports,
    )


def create_privacy_outputter(fmt: str, sink: OutputCallback) -> PrivacyReportOutputter:
    if fmt == "jsonl":
        return JsonLinesPrivacyReportOutputter(sink)
    raise ValueError(f"Unsupported output format '{fmt}'")
