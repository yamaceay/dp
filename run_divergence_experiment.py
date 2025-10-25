#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional

from experiments.divergence.io import (
    build_divergence_evaluation_inputs,
    build_original_texts,
    build_record_info,
)
from experiments.divergence.reporting import build_divergence_report, create_divergence_outputter
from experiments.utils import build_output_sink, collect_jsonl_sources
from dp.experiments.divergence import (
    BERTScoreDivergence,
    CosineSimilarityDivergence,
    TextDivergenceExperiment,
)
from dp.loaders import DatasetRecord, get_adapter


def load_dataset_records(
    dataset: str,
    data_in: Optional[str],
    max_records: Optional[int],
) -> List[DatasetRecord]:
    adapter = get_adapter(dataset, data=dataset, data_in=data_in, max_records=max_records)
    return list(adapter.iter_records())


def main() -> None:
    parser = argparse.ArgumentParser(description="Run semantic divergence experiment")
    parser.add_argument("--dataset", required=True, help="Dataset adapter name")
    parser.add_argument("--data_in", required=True, help="Dataset input path")
    parser.add_argument("--max_records", type=int, default=None, help="Maximum records to evaluate")
    parser.add_argument("--annotations_in", nargs="+", help="Directories or files containing anonymized JSONL outputs")
    parser.add_argument("--model_type", type=str, default=None, help="BERTScore model type")
    parser.add_argument("--language", type=str, default=None, help="Language code for baseline rescaling")
    parser.add_argument("--batch_size", type=int, default=16, help="BERTScore batch size")
    parser.add_argument("--device", type=str, default=None, help="Computation device for BERTScore")
    parser.add_argument("--rescale_with_baseline", action="store_true", help="Enable BERTScore baseline rescaling")
    parser.add_argument("--metric", type=str, default="bertscore", choices=["bertscore", "cosine"], help="Divergence metric to use")
    parser.add_argument("--output_format", type=str, default="text", choices=["text", "json", "jsonl"], help="Output format")
    parser.add_argument("--output_file", type=str, default=None, help="Path to output file")
    args = parser.parse_args()

    original_dataset = load_dataset_records(args.dataset, args.data_in, args.max_records)
    if not original_dataset:
        raise RuntimeError("No records loaded from dataset")
    if not args.annotations_in:
        raise RuntimeError("No anonymized outputs provided")
    sources: Dict[str, Path] = collect_jsonl_sources(*args.annotations_in)
    if not sources:
        raise RuntimeError("No anonymized output files discovered")
    evaluation_inputs: Dict[str, Dict[str, Any]] = build_divergence_evaluation_inputs(original_dataset, sources)
    evaluation_inputs = {
        name: payload for name, payload in evaluation_inputs.items() if payload.get("texts")
    }
    if not evaluation_inputs:
        raise RuntimeError("No anonymized outputs aligned with dataset records")
    record_info = build_record_info(original_dataset)
    original_texts = build_original_texts(original_dataset)
    if args.metric == "bertscore":
        experiment: TextDivergenceExperiment = BERTScoreDivergence(
            model_type=args.model_type,
            language=args.language,
            batch_size=args.batch_size,
            device=args.device,
            rescale_with_baseline=args.rescale_with_baseline,
        )
    else:
        experiment = CosineSimilarityDivergence()
    experiment.setup(
        original_texts=original_texts,
        evaluation_datasets=evaluation_inputs,
        record_info=record_info,
    )
    result = experiment.run()
    experiment.cleanup()

    report = build_divergence_report(
        result=result,
        original_record_count=len(original_dataset),
        evaluation_sources=sources,
    )
    sink = build_output_sink(args.output_file)
    outputter = create_divergence_outputter(args.output_format, sink)
    outputter.output(report)


if __name__ == "__main__":
    main()
