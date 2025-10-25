#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional

from experiments.privacy.io import build_privacy_evaluation_dataset
from experiments.privacy.reporting import build_privacy_report, create_privacy_outputter
from experiments.utils import build_output_sink, collect_jsonl_sources
from dp.experiments.privacy_annotations import TextPrivacyExperiment
from dp.loaders import DatasetRecord, get_adapter
from dp.loaders.annotations import read_batch_annotations_from_path


def load_dataset_records(
    dataset: str,
    data_in: Optional[str],
    max_records: Optional[int],
) -> List[DatasetRecord]:
    adapter = get_adapter(dataset, data=dataset, data_in=data_in, max_records=max_records)
    return list(adapter.iter_records())


def main() -> None:
    parser = argparse.ArgumentParser(description="Run annotation-driven privacy experiment")
    parser.add_argument("--dataset", required=True, help="Dataset adapter name")
    parser.add_argument("--data_in", required=True, help="Dataset input path")
    parser.add_argument("--max_records", type=int, default=None, help="Maximum records to evaluate")
    parser.add_argument("--tri_pipeline", required=True, help="Path to TRI pipeline directory")
    parser.add_argument("--tri_max_length", type=int, default=512, help="TRI maximum sequence length")
    parser.add_argument("--tri_device", type=str, default="auto", help="TRI device (auto, cpu, cuda, mps)")
    parser.add_argument("--annotations_in", nargs="+", help="Directories or files with annotation JSONL outputs")
    parser.add_argument("--mask_token", type=str, default="[MASK]", help="Token used to mask annotations")
    parser.add_argument("--output_format", type=str, default="text", choices=["text", "json", "jsonl"], help="Output format")
    parser.add_argument("--output_file", type=str, default=None, help="Path to output file (prints to stdout when omitted)")
    args = parser.parse_args()

    original_dataset = load_dataset_records(args.dataset, args.data_in, args.max_records)
    if not original_dataset:
        raise RuntimeError("No records loaded from dataset")
    original_record_count = len(original_dataset)
    if not args.annotations_in:
        raise RuntimeError("No annotation sources provided")
    annotation_sources: Dict[str, Path] = collect_jsonl_sources(*args.annotations_in)
    if not annotation_sources:
        raise RuntimeError("No annotation files discovered")
    evaluation_datasets: Dict[str, List[DatasetRecord]] = {}
    evaluation_record_counts: Dict[str, int] = {}
    for name, path in annotation_sources.items():
        annotations_batch = read_batch_annotations_from_path(str(path))
        evaluation_dataset = build_privacy_evaluation_dataset(original_dataset, annotations_batch, args.mask_token)
        evaluation_datasets[name] = evaluation_dataset
        evaluation_record_counts[name] = len(evaluation_dataset)

    experiment = TextPrivacyExperiment(
        tri_pipeline=args.tri_pipeline,
        tri_max_length=args.tri_max_length,
        tri_device=args.tri_device,
    )
    experiment.setup(
        dataset_name=args.dataset,
        original_dataset=original_dataset,
        evaluation_datasets=evaluation_datasets,
        progress=True,
    )
    result = experiment.run(progress=True)
    experiment.cleanup()

    report = build_privacy_report(
        result=result,
        original_record_count=original_record_count,
        annotation_sources=annotation_sources,
        evaluation_record_counts=evaluation_record_counts,
    )
    sink = build_output_sink(args.output_file)
    outputter = create_privacy_outputter(args.output_format, sink)
    outputter.output(report)


if __name__ == "__main__":
    main()
