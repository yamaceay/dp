#!/usr/bin/env python3

from __future__ import annotations

import argparse
from typing import Iterable, Optional

from experiments.semantic_divergence import SemanticDivergenceExperiment
from dp.loaders import get_adapter
from dp.loaders.base import DatasetRecord


def identity_anonymizer(text: str) -> str:
    return text


def collect_records(dataset: str, data_in: Optional[str], max_records: Optional[int]) -> Iterable[DatasetRecord]:
    adapter = get_adapter(dataset, data=dataset, data_in=data_in, max_records=max_records)
    return adapter.iter_records()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run semantic divergence experiment")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset adapter name")
    parser.add_argument("--data_in", type=str, default=None, help="Dataset input path")
    parser.add_argument("--max_records", type=int, default=None, help="Maximum records to evaluate")
    parser.add_argument("--model_type", type=str, default=None, help="BERTScore model type")
    parser.add_argument("--language", type=str, default=None, help="Language code for baseline rescaling")
    parser.add_argument("--batch_size", type=int, default=16, help="BERTScore batch size")
    parser.add_argument("--device", type=str, default=None, help="Computation device for BERTScore")
    parser.add_argument("--rescale_with_baseline", action="store_true", help="Enable BERTScore baseline rescaling")
    args = parser.parse_args()

    records = collect_records(args.dataset, args.data_in, args.max_records)
    experiment = SemanticDivergenceExperiment(
        records=records,
        anonymize=identity_anonymizer,
        max_records=args.max_records,
        model_type=args.model_type,
        language=args.language,
        batch_size=args.batch_size,
        device=args.device,
        rescale_with_baseline=args.rescale_with_baseline,
    )
    result = experiment.execute()
    print("Semantic Divergence")
    for key, value in result.metrics.items():
        print(f"  {key}: {value:.4f}")
    print(f"  score: {result.score:.4f}")


if __name__ == "__main__":
    main()
