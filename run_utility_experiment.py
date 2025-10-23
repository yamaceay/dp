#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from dp.experiments import ExperimentResult
from dp.experiments.utils import collect_jsonl_sources, uniquify_records, OutputCallback, build_output_sink
from dp.experiments.utility.base import TextUtilityExperiment
from dp.experiments.constants import UTILITY_EXPERIMENTS_REGISTRY 
from dp.loaders import DatasetRecord, get_adapter


@dataclass(frozen=True)
class UtilityEvaluationReport:
    name: str
    source: Optional[Path]
    valid: bool
    f1: Optional[float]
    loss: Optional[float]
    train_matched: int
    train_total: int
    test_matched: int
    test_total: int
    available: int


@dataclass(frozen=True)
class UtilityExperimentReport:
    score: float
    baseline_f1: float
    baseline_train_size: int
    baseline_test_size: int
    evaluations: List[UtilityEvaluationReport]


class ExperimentResultOutputter:
    def __init__(self, sink: OutputCallback):
        self.sink = sink

    def output(self, report: UtilityExperimentReport) -> None:
        raise NotImplementedError


class TextExperimentResultOutputter(ExperimentResultOutputter):
    def output(self, report: UtilityExperimentReport) -> None:
        lines: List[str] = []
        lines.append(f"Score: {report.score:.4f}")
        lines.append("")
        lines.append("Baseline")
        lines.append(
            f"  f1={report.baseline_f1:.4f} train={report.baseline_train_size} test={report.baseline_test_size}"
        )
        lines.append("")
        lines.append("Evaluation datasets")
        if not report.evaluations:
            lines.append("  none")
        else:
            for evaluation in report.evaluations:
                prefix = f"  {evaluation.name}: "
                if evaluation.valid and evaluation.f1 is not None and evaluation.loss is not None:
                    prefix += f"f1={evaluation.f1:.4f} loss={evaluation.loss:.4f}"
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


class JsonExperimentResultOutputter(ExperimentResultOutputter):
    def output(self, report: UtilityExperimentReport) -> None:
        payload: Dict[str, Any] = {
            "score": report.score,
            "baseline": {
                "f1": report.baseline_f1,
                "train_size": report.baseline_train_size,
                "test_size": report.baseline_test_size,
            },
            "evaluations": [
                {
                    "name": evaluation.name,
                    "source": str(evaluation.source) if evaluation.source else None,
                    "valid": evaluation.valid,
                    "f1": evaluation.f1,
                    "loss": evaluation.loss,
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


class JsonLinesExperimentResultOutputter(ExperimentResultOutputter):
    def output(self, report: UtilityExperimentReport) -> None:
        records: List[Dict[str, Any]] = [
            {
                "type": "experiment",
                "score": report.score,
                "baseline_f1": report.baseline_f1,
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
                    "f1": evaluation.f1,
                    "loss": evaluation.loss,
                    "train_matched": evaluation.train_matched,
                    "train_total": evaluation.train_total,
                    "test_matched": evaluation.test_matched,
                    "test_total": evaluation.test_total,
                    "available": evaluation.available,
                }
            )
        serialized = "\n".join(json.dumps(record, ensure_ascii=False) for record in records)
        self.sink(serialized)


def create_outputter(fmt: str, sink: OutputCallback) -> ExperimentResultOutputter:
    if fmt == "text":
        return TextExperimentResultOutputter(sink)
    if fmt == "json":
        return JsonExperimentResultOutputter(sink)
    if fmt == "jsonl":
        return JsonLinesExperimentResultOutputter(sink)
    raise ValueError(f"Unsupported output format '{fmt}'")


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    entries: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                entries.append(json.loads(text))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number} in {path}") from exc
    return entries

def build_evaluation_texts(index_to_key: Dict[int, str], sources: Dict[str, Path]) -> Dict[str, Dict[str, str]]:
    evaluations: Dict[str, Dict[str, str]] = {}
    for name, path in sources.items():
        mapping: Dict[str, str] = {}
        for entry in read_jsonl(path):
            idx = entry.get("idx")
            text = entry.get("text", "")
            if idx is None or not text:
                continue
            try:
                idx_value = int(idx)
            except (TypeError, ValueError):
                continue
            key = index_to_key.get(idx_value)
            if not key:
                continue
            mapping[key] = text
        evaluations[name] = mapping
    return evaluations


def build_experiment_report(result: ExperimentResult, sources: Dict[str, Path]) -> UtilityExperimentReport:
    metrics = result.metrics or {}
    baseline = metrics.get("baseline", {})
    baseline_f1 = float(baseline.get("f1", 0.0))
    baseline_train = int(baseline.get("train_size", 0))
    baseline_test = int(baseline.get("test_size", 0))
    evaluation_metrics: Dict[str, Dict[str, Any]] = metrics.get("evaluations", {})
    evaluations: List[UtilityEvaluationReport] = []
    for name in sorted(evaluation_metrics.keys()):
        payload = evaluation_metrics.get(name) or {}
        evaluations.append(
            UtilityEvaluationReport(
                name=name,
                source=sources.get(name),
                valid=bool(payload.get("valid")),
                f1=payload.get("f1"),
                loss=payload.get("loss"),
                train_matched=int(payload.get("train_matched", 0)),
                train_total=int(payload.get("train_total", 0)),
                test_matched=int(payload.get("test_matched", 0)),
                test_total=int(payload.get("test_total", 0)),
                available=int(payload.get("available", 0)),
            )
        )
    return UtilityExperimentReport(
        score=result.score,
        baseline_f1=baseline_f1,
        baseline_train_size=baseline_train,
        baseline_test_size=baseline_test,
        evaluations=evaluations,
    )

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run text utility experiment")
    parser.add_argument("--dataset", required=True, choices=["reddit", "tab", "db_bio", "trustpilot"], help="Dataset name")
    parser.add_argument("--data_in", required=True, help="Path to dataset input")
    parser.add_argument("--annotations_in", nargs="+", help="Directories or files containing anonymized outputs")
    parser.add_argument("--max_records", type=int, default=None, help="Maximum records to use")
    parser.add_argument("--test_size", type=float, default=0.2, help="Test split fraction")
    parser.add_argument("--random_state", type=int, default=42, help="Random seed for splits")
    parser.add_argument("--target", required=True, help="Experiment target")
    parser.add_argument("--output_format", choices=["text", "json", "jsonl"], default="text", help="Output format")
    parser.add_argument("--output_file", default=None, help="Optional report path")
    parser.add_argument("--dry_run", action="store_true", help="Print dataset summary and exit")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw_records = get_adapter(args.dataset, data=args.dataset, data_in=args.data_in, max_records=args.max_records)
    if not raw_records:
        raise RuntimeError("No records loaded from dataset")
    records = uniquify_records(raw_records)
    experiment_adapter = UTILITY_EXPERIMENTS_REGISTRY.get(f"{args.dataset}_{args.target}")
    if not args.annotations_in:
        raise RuntimeError("No anonymized output files provided")
    index_to_key = {idx: record.uid for idx, record in enumerate(records)}
    sources = collect_jsonl_sources(*args.annotations_in)
    if not sources:
        raise RuntimeError("No anonymized output files discovered")
    evaluation_texts = build_evaluation_texts(index_to_key, sources)
    evaluation_texts = {name: mapping for name, mapping in evaluation_texts.items() if mapping}
    if not evaluation_texts:
        raise RuntimeError("No anonymized texts aligned with dataset records")
    experiment_adapter.setup(target=args.target, records=records, model=model)
    result = experiment_adapter.run(evaluation_texts=evaluation_texts)
    experiment_adapter.cleanup()
    report = build_experiment_report(result, sources)
    sink = build_output_sink(args.output_file)
    outputter = create_outputter(args.output_format, sink)
    outputter.output(report)


if __name__ == "__main__":
    main()
