#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from dp.experiments import ExperimentResult
from dp.experiments.utils import collect_jsonl_sources, uniquify_records, OutputCallback, build_output_sink
from dp.experiments.semantic_divergence import TextDivergenceExperiment
from dp.loaders import DatasetRecord, get_adapter


@dataclass(frozen=True)
class DivergenceEntry:
    key: str
    index: int
    name: Optional[str]
    persona_uid: Optional[str]
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
    original_record_count: int
    evaluations: List[DivergenceEvaluationReport]


class ExperimentResultOutputter(ABC):
    def __init__(self, sink: OutputCallback):
        self.sink = sink

    @abstractmethod
    def output(self, report: DivergenceExperimentReport) -> None:
        raise NotImplementedError()


class TextExperimentResultOutputter(ExperimentResultOutputter):
    def output(self, report: DivergenceExperimentReport) -> None:
        lines: List[str] = []
        lines.append(f"Score: {report.score:.4f}")
        lines.append("")
        lines.append("Original dataset")
        lines.append(f"  records: {report.original_record_count}")
        lines.append("")
        lines.append("Evaluation datasets")
        if not report.evaluations:
            lines.append("  none")
        else:
            for evaluation in report.evaluations:
                base = f"  {evaluation.name}: matched {evaluation.matched_count}/{report.original_record_count}"
                base += f" records"
                if evaluation.available_count and evaluation.available_count != evaluation.matched_count:
                    base += f" ({evaluation.available_count} available)"
                if evaluation.source:
                    base += f" from {evaluation.source}"
                lines.append(base)
                summary = evaluation.summary or {}
                if summary:
                    lines.append(
                        "    divergence: "
                        f"mean={summary.get('divergence_mean', 0.0):.4f} "
                        f"median={summary.get('divergence_median', 0.0):.4f} "
                        f"min={summary.get('divergence_min', 0.0):.4f} "
                        f"max={summary.get('divergence_max', 0.0):.4f}"
                    )
                    lines.append(
                        "    similarity: "
                        f"mean={summary.get('similarity_mean', 0.0):.4f} "
                        f"median={summary.get('similarity_median', 0.0):.4f} "
                        f"min={summary.get('similarity_min', 0.0):.4f} "
                        f"max={summary.get('similarity_max', 0.0):.4f}"
                    )
                for entry in evaluation.entries:
                    suffix = f" ({entry.name})" if entry.name else ""
                    lines.append(
                        f"    [{entry.index}] {entry.key}{suffix}: "
                        f"divergence={entry.divergence:.4f} similarity={entry.similarity:.4f}"
                    )
        self.sink("\n".join(lines))


class JsonExperimentResultOutputter(ExperimentResultOutputter):
    def output(self, report: DivergenceExperimentReport) -> None:
        payload: Dict[str, Any] = {
            "score": report.score,
            "original": {
                "record_count": report.original_record_count,
            },
            "evaluations": [
                {
                    "name": evaluation.name,
                    "matched_count": evaluation.matched_count,
                    "available_count": evaluation.available_count,
                    "source": str(evaluation.source) if evaluation.source else None,
                    "summary": evaluation.summary,
                    "entries": [
                        {
                            "key": entry.key,
                            "index": entry.index,
                            "name": entry.name,
                            "persona_uid": entry.persona_uid,
                            "similarity": entry.similarity,
                            "divergence": entry.divergence,
                        }
                        for entry in evaluation.entries
                    ],
                }
                for evaluation in report.evaluations
            ],
        }
        self.sink(json.dumps(payload, ensure_ascii=False, indent=2))


class JsonLinesExperimentResultOutputter(ExperimentResultOutputter):
    def output(self, report: DivergenceExperimentReport) -> None:
        records: List[Dict[str, Any]] = []
        records.append(
            {
                "type": "experiment",
                "score": report.score,
                "original_record_count": report.original_record_count,
            }
        )
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
                        "persona_uid": entry.persona_uid,
                        "similarity": entry.similarity,
                        "divergence": entry.divergence,
                    }
                )
        serialized = "\n".join(json.dumps(record, ensure_ascii=False) for record in records)
        self.sink(serialized)


def build_experiment_report(
    result: ExperimentResult,
    original_record_count: int,
    evaluation_sources: Dict[str, Path],
) -> DivergenceExperimentReport:
    metrics = result.metrics or {}
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
        entries: List[DivergenceEntry] = []
        def sort_key(item: str) -> int:
            info = record_info.get(item, {})
            return int(info.get("index", 0))
        for key in sorted(similarities.keys(), key=sort_key):
            info = record_info.get(key, {})
            entries.append(
                DivergenceEntry(
                    key=key,
                    index=int(info.get("index", 0)),
                    name=info.get("name"),
                    persona_uid=info.get("persona_uid"),
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
        score=result.score,
        original_record_count=original_record_count,
        evaluations=evaluations,
    )


def create_outputter(fmt: str, sink: OutputCallback) -> ExperimentResultOutputter:
    if fmt == "text":
        return TextExperimentResultOutputter(sink)
    if fmt == "json":
        return JsonExperimentResultOutputter(sink)
    if fmt == "jsonl":
        return JsonLinesExperimentResultOutputter(sink)
    raise ValueError(f"Unsupported output format '{fmt}'")


def load_dataset_records(
    dataset: str,
    data_in: Optional[str],
    max_records: Optional[int],
) -> List[DatasetRecord]:
    adapter = get_adapter(dataset, data=dataset, data_in=data_in, max_records=max_records)
    return list(adapter.iter_records())


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for index, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                items.append(json.loads(text))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {index} in {path}") from exc
    return items


def build_evaluation_inputs(
    original_dataset: List[DatasetRecord],
    sources: Dict[str, Path],
) -> Dict[str, Dict[str, Any]]:
    index_to_key = {idx: record.uid for idx, record in enumerate(original_dataset)}
    evaluations: Dict[str, Dict[str, Any]] = {}
    for name, path in sources.items():
        records = read_jsonl(path)
        texts: Dict[str, str] = {}
        for entry in records:
            idx = entry.get("idx")
            text = entry.get("text", "")
            if idx is None:
                continue
            key = index_to_key.get(int(idx))
            if not key or not text:
                continue
            texts[key] = text
        evaluations[name] = {
            "texts": texts,
            "total": len(records),
        }
    return evaluations


def build_record_info(records: List[DatasetRecord]) -> Dict[str, Dict[str, Any]]:
    info: Dict[str, Dict[str, Any]] = {}
    for index, record in enumerate(records, start=1):
        metadata = record.metadata or {}
        info[record.uid] = {
            "index": metadata.get("record_index", index),
            "name": record.name,
            "persona_uid": metadata.get("persona_uid"),
        }
    return info


def build_original_texts(records: List[DatasetRecord]) -> Dict[str, str]:
    return {record.uid: record.text for record in records}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run semantic divergence experiment")
    parser.add_argument("--dataset", required=True, help="Dataset adapter name")
    parser.add_argument("--data_in", required=True, help="Dataset input path")
    parser.add_argument("--max_records", type=int, default=None, help="Maximum records to evaluate")
    parser.add_argument("--annotations_in", nargs="+", help="One or more directories or files containing anonymized JSONL outputs")
    parser.add_argument("--model_type", type=str, default=None, help="BERTScore model type")
    parser.add_argument("--language", type=str, default=None, help="Language code for baseline rescaling")
    parser.add_argument("--batch_size", type=int, default=16, help="BERTScore batch size")
    parser.add_argument("--device", type=str, default=None, help="Computation device for BERTScore")
    parser.add_argument("--rescale_with_baseline", action="store_true", help="Enable BERTScore baseline rescaling")
    parser.add_argument("--metric", type=str, default="bertscore", choices=["bertscore"], help="Divergence metric to use")
    parser.add_argument("--output_format", type=str, default="text", choices=["text", "json", "jsonl"], help="Output format")
    parser.add_argument("--output_file", type=str, default=None, help="Path to output file (if omitted, prints to stdout)")
    args = parser.parse_args()

    raw_original_dataset = load_dataset_records(args.dataset, args.data_in, args.max_records)
    if not raw_original_dataset:
        raise RuntimeError("No records loaded from dataset")
    original_dataset = uniquify_records(raw_original_dataset)
    original_record_count = len(original_dataset)
    if not args.annotations_in:
        raise RuntimeError("No anonymized outputs provided")
    sources = collect_jsonl_sources(*args.annotations_in)
    if not sources:
        raise RuntimeError("No anonymized output files discovered")
    evaluation_inputs = build_evaluation_inputs(original_dataset, sources)
    evaluation_inputs = {
        name: payload for name, payload in evaluation_inputs.items() if payload.get("texts")
    }
    if not evaluation_inputs:
        raise RuntimeError("No anonymized outputs aligned with dataset records")
    record_info = build_record_info(original_dataset)
    original_texts = build_original_texts(original_dataset)
    experiment = TextDivergenceExperiment(
        original_texts=original_texts,
        evaluation_datasets=evaluation_inputs,
        record_info=record_info,
        model_type=args.model_type,
        language=args.language,
        batch_size=args.batch_size,
        device=args.device,
        rescale_with_baseline=args.rescale_with_baseline,
        metric=args.metric,
    )
    experiment.setup()
    result = experiment.run()
    experiment.cleanup()
    report = build_experiment_report(
        result=result,
        original_record_count=original_record_count,
        evaluation_sources=sources,
    )
    sink = build_output_sink(args.output_file)
    outputter = create_outputter(args.output_format, sink)
    outputter.output(report)


if __name__ == "__main__":
    main()
