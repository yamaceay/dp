#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from experiments import ExperimentResult
from experiments.privacy_annotations import AnnotationPrivacyExperiment
from loaders import DatasetRecord, TextAnnotation, get_adapter
from loaders.annotations import apply_annotations, read_batch_annotations_from_path


@dataclass(frozen=True)
class RankEntry:
    key: str
    rank: int
    name: Optional[str]
    persona_uid: Optional[str]
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
class ExperimentReport:
    score: float
    original_record_count: int
    original_ranks: List[RankEntry]
    evaluations: List[EvaluationReport]


OutputCallback = Callable[[str], None]


class ExperimentResultOutputter(ABC):
    def __init__(self, sink: OutputCallback):
        self.sink = sink

    @abstractmethod
    def output(self, report: ExperimentReport) -> None:
        raise NotImplementedError()


class TextExperimentResultOutputter(ExperimentResultOutputter):
    def output(self, report: ExperimentReport) -> None:
        lines: List[str] = []
        lines.append(f"Score: {report.score:.4f}")
        lines.append("")
        lines.append("Original dataset")
        lines.append(f"  records: {report.original_record_count}")
        lines.append("")
        lines.append("Original ranks")
        if report.original_ranks:
            for entry in report.original_ranks:
                prefix = f"[{entry.index}] {entry.key}"
                suffix = f" ({entry.name})" if entry.name else ""
                lines.append(f"  {prefix}{suffix}: {entry.rank}")
        else:
            lines.append("  none")
        lines.append("")
        lines.append("Evaluation datasets")
        if not report.evaluations:
            lines.append("  none")
        else:
            for evaluation in report.evaluations:
                description = f"  {evaluation.name}: {evaluation.record_count} records"
                if evaluation.source:
                    description += f" from {evaluation.source}"
                lines.append(description)
                summary = evaluation.summary
                if summary:
                    lines.append(
                        "    summary: "
                        f"count={summary.get('count', 0)} "
                        f"improved={summary.get('improved', 0)} "
                        f"degraded={summary.get('degraded', 0)} "
                        f"unchanged={summary.get('unchanged', 0)} "
                        f"mean={summary.get('mean', 0.0):.4f} "
                        f"median={summary.get('median', 0.0):.4f} "
                        f"min={summary.get('min', 0)} "
                        f"max={summary.get('max', 0)}"
                    )
                for entry in evaluation.ranks:
                    prefix = f"[{entry.index}] {entry.key}"
                    suffix = f" ({entry.name})" if entry.name else ""
                    if entry.delta is not None:
                        lines.append(f"    {prefix}{suffix}: {entry.rank} delta={entry.delta:+d}")
                    else:
                        lines.append(f"    {prefix}{suffix}: {entry.rank}")
        self.sink("\n".join(lines))


class JsonExperimentResultOutputter(ExperimentResultOutputter):
    def output(self, report: ExperimentReport) -> None:
        payload: Dict[str, Any] = {
            "score": report.score,
            "original": {
                "record_count": report.original_record_count,
                "ranks": [self._serialize_rank(entry) for entry in report.original_ranks],
            },
            "evaluations": [
                {
                    "name": evaluation.name,
                    "record_count": evaluation.record_count,
                    "source": str(evaluation.source) if evaluation.source else None,
                    "summary": evaluation.summary,
                    "ranks": [self._serialize_rank(entry) for entry in evaluation.ranks],
                }
                for evaluation in report.evaluations
            ],
        }
        self.sink(json.dumps(payload, ensure_ascii=False, indent=2))

    @staticmethod
    def _serialize_rank(entry: RankEntry) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "key": entry.key,
            "rank": entry.rank,
            "index": entry.index,
        }
        if entry.persona_uid:
            payload["persona_uid"] = entry.persona_uid
        if entry.name:
            payload["name"] = entry.name
        if entry.delta is not None:
            payload["delta"] = entry.delta
        return payload


class JsonLinesExperimentResultOutputter(ExperimentResultOutputter):
    def output(self, report: ExperimentReport) -> None:
        records: List[Dict[str, Any]] = []
        records.append(
            {
                "type": "experiment",
                "score": report.score,
                "original_record_count": report.original_record_count,
            }
        )
        for entry in report.original_ranks:
            records.append(self._rank_record("original_rank", entry))
        for evaluation in report.evaluations:
            records.append(
                {
                    "type": "evaluation",
                    "name": evaluation.name,
                    "record_count": evaluation.record_count,
                    "source": str(evaluation.source) if evaluation.source else None,
                    "summary": evaluation.summary,
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
        if entry.persona_uid:
            payload["persona_uid"] = entry.persona_uid
        if entry.name:
            payload["name"] = entry.name
        if entry.delta is not None:
            payload["delta"] = entry.delta
        if evaluation:
            payload["evaluation"] = evaluation
        return payload


def build_experiment_report(
    result: ExperimentResult,
    original_record_count: int,
    annotation_sources: Dict[str, Path],
    evaluation_record_counts: Dict[str, int],
) -> ExperimentReport:
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
                persona_uid=info.get("persona_uid"),
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
                    persona_uid=info.get("persona_uid"),
                    index=int(info.get("index", 0)),
                    delta=deltas.get(key),
                )
            )
        evaluation_reports.append(
            EvaluationReport(
                name=name,
                source=annotation_sources.get(name),
                record_count=evaluation_record_counts.get(name, 0),
                ranks=ranks,
                summary=payload.get("rank_delta_summary"),
            )
        )
    return ExperimentReport(
        score=result.score,
        original_record_count=original_record_count,
        original_ranks=original_ranks,
        evaluations=evaluation_reports,
    )


def build_output_sink(output_file: Optional[str]) -> OutputCallback:
    if not output_file:
        return print
    path = Path(output_file).expanduser()
    def sink(content: str) -> None:
        normalized = content if content.endswith("\n") else f"{content}\n"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(normalized, encoding="utf-8")
        print(f"Report written to {path}")
    return sink


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

def build_evaluation_dataset(
    records: List[DatasetRecord],
    annotations_batch: List[List[TextAnnotation]],
    mask_token: str,
) -> List[DatasetRecord]:
    dataset: List[DatasetRecord] = []
    total = len(annotations_batch)
    for index, record in enumerate(records):
        annotations = annotations_batch[index] if index < total else []
        text = apply_annotations(record.text, annotations, replacement=mask_token)
        dataset.append(
            DatasetRecord(
                text=text,
                uid=record.uid,
                name=record.name,
                spans=record.spans,
                utilities=dict(record.utilities),
                metadata=dict(record.metadata),
            )
        )
    return dataset


def uniquify_records(records: List[DatasetRecord]) -> List[DatasetRecord]:
    counts: Dict[str, int] = {}
    unique_records: List[DatasetRecord] = []
    for index, record in enumerate(records, start=1):
        original_uid = record.uid or ""
        base_uid = original_uid if original_uid else f"record_{index}"
        occurrence = counts.get(base_uid, 0)
        counts[base_uid] = occurrence + 1
        unique_uid = base_uid if occurrence == 0 else f"{base_uid}#{occurrence}"
        metadata = dict(record.metadata)
        persona_uid = metadata.get("persona_uid", original_uid)
        if not persona_uid:
            persona_uid = base_uid
        metadata["persona_uid"] = persona_uid
        metadata["record_index"] = index
        spans = list(record.spans) if record.spans else None
        unique_records.append(
            DatasetRecord(
                text=record.text,
                uid=unique_uid,
                name=record.name,
                spans=spans,
                utilities=dict(record.utilities),
                metadata=metadata,
            )
        )
    return unique_records


def collect_annotation_sources(
    *paths: str,
) -> Dict[str, Path]:
    paths = [Path(p) for p in paths]
    entries: List[Tuple[str, Path]] = []
    counts: Dict[str, int] = {}
    seen: set[Path] = set()

    def register(path: Path) -> None:
        if path in seen:
            return
        seen.add(path)
        dataset_name = path.parent.parent.stem
        method_name = path.parent.stem
        base = f"{method_name}_{dataset_name}"
        occurrence = counts.get(base, 0)
        counts[base] = occurrence + 1
        name = f"{base}_{occurrence}"
        entries.append((name, path))

    for path in paths:
        if path.is_file():
            register(path)
        elif path.is_dir():
            for file_path in sorted(path.rglob("*.jsonl")):
                register(file_path)

    return {name: entry_path for name, entry_path in entries}

def main() -> None:
    parser = argparse.ArgumentParser(description="Run annotation-driven privacy experiment")
    parser.add_argument("--dataset", required=True, help="Dataset adapter name")
    parser.add_argument("--data_in", required=True, help="Dataset input path")
    parser.add_argument("--max_records", type=int, default=None, help="Maximum records to evaluate")
    parser.add_argument("--tri_pipeline", required=True, help="Path to TRI pipeline directory")
    parser.add_argument("--tri_max_length", type=int, default=512, help="TRI maximum sequence length")
    parser.add_argument("--tri_device", type=str, default="auto", help="TRI device (auto, cpu, cuda, mps)")
    parser.add_argument("--annotations_in", nargs='+', help="One or more directories or files to search recursively for annotation files")
    parser.add_argument("--mask_token", type=str, default="[MASK]", help="Token used to mask annotations")
    parser.add_argument("--output_format", type=str, default="text", choices=["text", "json", "jsonl"], help="Output format")
    parser.add_argument("--output_file", type=str, default=None, help="Path to output file (if not specified, prints to stdout)")
    args = parser.parse_args()

    raw_original_dataset = load_dataset_records(args.dataset, args.data_in, args.max_records)
    if not raw_original_dataset:
        raise RuntimeError("No records loaded from dataset")
    original_dataset = uniquify_records(raw_original_dataset)
    original_record_count = len(original_dataset)

    evaluation_datasets: Dict[str, List[DatasetRecord]] = {}
    annotation_sources: Dict[str, Path] = {}
    evaluation_record_counts: Dict[str, int] = {}
    if not args.annotations_in:
        raise RuntimeError("No annotation sources provided")
    annotations_found = collect_annotation_sources(*args.annotations_in)
    if not annotations_found:
        raise RuntimeError("No annotation files discovered")
    for name, path in annotations_found.items():
        annotations_batch = read_batch_annotations_from_path(str(path))
        evaluation_dataset = build_evaluation_dataset(original_dataset, annotations_batch, args.mask_token)
        evaluation_datasets[name] = evaluation_dataset
        annotation_sources[name] = path
        evaluation_record_counts[name] = len(evaluation_dataset)

    experiment = AnnotationPrivacyExperiment(
        dataset_name=args.dataset,
        original_dataset=original_dataset,
        evaluation_datasets=evaluation_datasets,
        tri_pipeline=args.tri_pipeline,
        tri_max_length=args.tri_max_length,
        tri_device=args.tri_device,
    )

    experiment.setup(progress=True)
    result = experiment.run(progress=True)
    experiment.cleanup()

    report = build_experiment_report(
        result=result,
        original_record_count=original_record_count,
        annotation_sources=annotation_sources,
        evaluation_record_counts=evaluation_record_counts,
    )
    sink = build_output_sink(args.output_file)
    outputter = create_outputter(args.output_format, sink)
    outputter.output(report)

if __name__ == "__main__":
    main()
