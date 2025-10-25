#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional

from experiments.utility.models import UTILITY_EXPERIMENTS_REGISTRY, UtilitySpec
from experiments.utility.io import build_utility_evaluation_texts
from experiments.utility.reporting import build_utility_report, create_utility_outputter
from experiments.utils import build_output_sink, collect_jsonl_sources
from dp.experiments.utility.base import TextUtilityExperiment
from dp.loaders import DatasetRecord, get_adapter


def load_dataset_records(
    dataset: str,
    data_in: Optional[str],
    max_records: Optional[int],
) -> List[DatasetRecord]:
    adapter = get_adapter(dataset, data=dataset, data_in=data_in, max_records=max_records)
    return list(adapter.iter_records())


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
    records = load_dataset_records(args.dataset, args.data_in, args.max_records)
    if not records:
        raise RuntimeError("No records loaded from dataset")
    spec_key = f"{args.dataset}_{args.target}"
    spec: Optional[UtilitySpec] = UTILITY_EXPERIMENTS_REGISTRY.get(spec_key)
    if spec is None:
        dataset_prefix = f"{args.dataset}_"
        available = sorted(
            key[len(dataset_prefix) :]
            for key in UTILITY_EXPERIMENTS_REGISTRY.keys()
            if key.startswith(dataset_prefix)
        )
        raise RuntimeError(
            f"Unknown utility target '{args.target}' for dataset '{args.dataset}'. "
            f"Available targets: {', '.join(available)}"
        )
    if args.dry_run:
        coverage = sum(1 for record in records if spec.target.value(record) is not None and record.text)
        print(f"Records loaded: {len(records)}")
        print(f"Target coverage: {coverage}")
        return
    if not args.annotations_in:
        raise RuntimeError("No anonymized output files provided")
    index_to_key = {idx: record.uid for idx, record in enumerate(records)}
    sources: Dict[str, Path] = collect_jsonl_sources(*args.annotations_in)
    if not sources:
        raise RuntimeError("No anonymized output files discovered")
    evaluation_texts: Dict[str, Dict[str, str]] = build_utility_evaluation_texts(index_to_key, sources)
    evaluation_texts = {name: mapping for name, mapping in evaluation_texts.items() if mapping}
    if not evaluation_texts:
        raise RuntimeError("No anonymized texts aligned with dataset records")
    experiment = TextUtilityExperiment(test_size=args.test_size, random_state=args.random_state)
    model = spec.build_model()
    experiment.setup(target=spec.target, records=records, model=model)
    result = experiment.run(evaluation_texts=evaluation_texts)
    experiment.cleanup()

    report = build_utility_report(result, sources)
    sink = build_output_sink(args.output_file)
    outputter = create_utility_outputter(args.output_format, sink)
    outputter.output(report)


if __name__ == "__main__":
    main()
