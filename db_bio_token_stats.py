from __future__ import annotations

import argparse
import math
import statistics
from pathlib import Path
from typing import Iterable, List, Sequence

from datasets import load_from_disk
from transformers import AutoTokenizer


def _iter_texts(data_root: Path, split_names: Sequence[str], field: str) -> Iterable[str]:
    for split_name in split_names:
        split_path = data_root / split_name
        if not split_path.exists():
            raise FileNotFoundError(f"Missing split path: {split_path}")
        dataset = load_from_disk(str(split_path))
        if field not in dataset.column_names:
            raise ValueError(f"Field '{field}' not found in split '{split_name}'. Available: {dataset.column_names}")
        for value in dataset[field]:
            if value is None:
                yield ""
            elif isinstance(value, str):
                yield value
            else:
                yield str(value)


def _percentile(sorted_values: List[int], q: float) -> float:
    if not sorted_values:
        return math.nan
    if q <= 0:
        return float(sorted_values[0])
    if q >= 1:
        return float(sorted_values[-1])
    index = (len(sorted_values) - 1) * q
    lower = int(math.floor(index))
    upper = int(math.ceil(index))
    if lower == upper:
        return float(sorted_values[lower])
    weight = index - lower
    return float(sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute DB-Bio token-length statistics")
    parser.add_argument("--data-root", type=Path, default=Path("data/db_bio"))
    parser.add_argument("--tokenizer", type=str, default="distilbert-base-uncased")
    parser.add_argument("--field", type=str, default="text")
    parser.add_argument("--splits", nargs="+", default=["train", "validation", "test"])
    parser.add_argument("--threshold", type=int, default=256)
    parser.add_argument("--add-special-tokens", action="store_true")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    token_lengths: List[int] = []
    for text in _iter_texts(args.data_root, args.splits, args.field):
        encoded = tokenizer(text, add_special_tokens=args.add_special_tokens, truncation=False)
        token_lengths.append(len(encoded["input_ids"]))

    if not token_lengths:
        raise RuntimeError("No records found")

    token_lengths_sorted = sorted(token_lengths)
    n = len(token_lengths)
    count_exceed = sum(1 for length in token_lengths if length > args.threshold)

    print(f"tokenizer={args.tokenizer}")
    print(f"data_root={args.data_root}")
    print(f"splits={','.join(args.splits)}")
    print(f"field={args.field}")
    print(f"add_special_tokens={args.add_special_tokens}")
    print(f"records={n}")
    print(f"threshold={args.threshold}")
    print(f"count_gt_threshold={count_exceed}")
    print(f"pct_gt_threshold={100.0 * count_exceed / n:.2f}")
    print(f"min={min(token_lengths)}")
    print(f"max={max(token_lengths)}")
    print(f"mean={statistics.fmean(token_lengths):.2f}")
    print(f"median={statistics.median(token_lengths):.2f}")
    print(f"std={statistics.pstdev(token_lengths):.2f}")
    print(f"p90={_percentile(token_lengths_sorted, 0.90):.2f}")
    print(f"p95={_percentile(token_lengths_sorted, 0.95):.2f}")
    print(f"p99={_percentile(token_lengths_sorted, 0.99):.2f}")


if __name__ == "__main__":
    main()