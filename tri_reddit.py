#!/usr/bin/env python3

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import List, Tuple

from sklearn.model_selection import train_test_split

from dp.loaders import get_adapter, DatasetRecord
from dp.utils import TRIDetector


def split_records(records: List[DatasetRecord], test_size: float, random_state: int) -> Tuple[List[DatasetRecord], List[DatasetRecord]]:
    if not records:
        raise ValueError("No records available for TRI")
    if len(records) < 2:
        raise ValueError("Need at least two records for train/test split")
    train, test = train_test_split(records, test_size=test_size, random_state=random_state, stratify=[r.name for r in records])
    return list(train), list(test)


def load_records(data_path: str, max_records: int | None) -> List[DatasetRecord]:
    adapter = get_adapter("reddit", data_in=data_path, max_records=max_records)
    return list(adapter.iter_records())


def train(args: argparse.Namespace) -> None:
    records = load_records(args.data_path, args.max_records)
    train_records, val_records = split_records(records, args.validation_split, args.random_state)
    detector = TRIDetector(
        dataset_name="reddit",
        model_name=args.model_name,
        max_length=args.max_length,
        device=args.device,
        use_chunking=True,
    )
    detector.set_train_dataset(train_records)
    detector.set_eval_datasets({"validation": val_records})
    detector.train(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        output_dir=args.output_dir,
        use_pretraining=args.use_pretraining,
        pretraining_epochs=args.pretraining_epochs,
        best_metric_dataset="validation",
    )


def evaluate(args: argparse.Namespace) -> None:
    records = load_records(args.data_path, args.max_records)
    detector = TRIDetector(
        dataset_name="reddit",
        model_name=args.model_name,
        max_length=args.max_length,
        device=args.device,
        use_chunking=True,
    )
    detector.load(args.model_path)
    metrics = detector.evaluate(records)
    print("Evaluation")
    for key, value in metrics.items():
        print(f"  {key}: {value}")


def predict(args: argparse.Namespace) -> None:
    records = load_records(args.data_path, args.max_records)
    detector = TRIDetector(
        dataset_name="reddit",
        model_name=args.model_name,
        max_length=args.max_length,
        device=args.device,
        use_chunking=True,
    )
    detector.load(args.model_path)
    predictions = detector.predict(records[: args.sample_count])
    print("Predictions")
    for uid, probs in predictions.items():
        top_label, score = max(probs.items(), key=lambda item: item[1])
        print(f"  {uid}: {top_label} ({score:.4f})")


def resolve_output_dir(base_dir: str | None) -> str:
    if base_dir:
        return base_dir
    return os.path.join("models", "tri_pipelines", "reddit")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="TRI workflow for Reddit dataset")
    parser.add_argument("--data-path", type=str, default="data/reddit/train.jsonl")
    parser.add_argument("--max-records", type=int, default=None)
    parser.add_argument("--model-name", type=str, default="distilbert-base-uncased")
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--mode", choices=["train", "evaluate", "predict"], default="train")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--use-pretraining", action="store_true")
    parser.add_argument("--pretraining-epochs", type=int, default=3)
    parser.add_argument("--validation-split", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--sample-count", type=int, default=5)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.output_dir = resolve_output_dir(args.output_dir)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    if args.mode == "train":
        train(args)
    elif args.mode == "evaluate":
        if not args.model_path:
            raise ValueError("model_path is required for evaluation")
        evaluate(args)
    else:
        if not args.model_path:
            raise ValueError("model_path is required for prediction")
        predict(args)


if __name__ == "__main__":
    main()
