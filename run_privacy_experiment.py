#!/usr/bin/env python3

from __future__ import annotations

import argparse
from typing import Iterable, Optional

from dp.experiments.reidentification import ReidentificationRiskExperiment
from dp.loaders import get_adapter
from dp.loaders.base import DatasetRecord


def identity_anonymizer(text: str) -> str:
    return text


def collect_records(dataset: str, data_in: Optional[str], max_records: Optional[int]) -> Iterable[DatasetRecord]:
    adapter = get_adapter(dataset, data=dataset, data_in=data_in, max_records=max_records)
    return adapter.iter_records()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run privacy re-identification experiment")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset adapter name")
    parser.add_argument("--data_in", type=str, default=None, help="Dataset input path")
    parser.add_argument("--max_records", type=int, default=None, help="Maximum records to evaluate")
    args = parser.parse_args()

    records = collect_records(args.dataset, args.data_in, args.max_records)
    experiment = ReidentificationRiskExperiment(
        records=records,
        anonymize=identity_anonymizer,
        max_records=args.max_records,
    )
    result = experiment.execute()
    print("Reidentification Privacy")
    for key, value in result.metrics.items():
        print(f"  {key}: {value:.4f}")
    print(f"  score: {result.score:.4f}")


if __name__ == "__main__":
    main()
