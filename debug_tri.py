import argparse
import json
from typing import Dict, List, Optional

from dp.loaders import get_adapter
from dp.loaders.results import build_dataset_from_results
from dp.tri.base import TRIDetector
from dp.utils.tasking import apply_task_template, resolve_task_id
from runtime.config_loader import _read_yaml


def normalize_config(config: Dict, args: argparse.Namespace) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=None)
    parser.add_argument('--data', type=str, default=None)
    parser.add_argument('--data_in', type=str, default=None)
    parser.add_argument('--split', type=str, default=None)
    parser.add_argument('--result_in', type=str, default=None)
    parser.add_argument('--pipeline_in', type=str, default=None)
    parser.add_argument('--risk_in', type=str, default=None)
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--end', type=int, default=None)
    parser.add_argument('--step', type=int, default=1)
    parser.add_argument('--max_records', type=int, default=None)
    parser.add_argument('--full_record', action='store_true')
    parser.add_argument('--save_to_jsonl', type=str, default=None)
    parser.add_argument('--n_first_predictions', type=int, default=0)

    new_args = parser.parse_args([])
    for key, value in config.items():
        if hasattr(new_args, key):
            setattr(new_args, key, value)
    for key, value in vars(args).items():
        if value is not None:
            setattr(new_args, key, value)
    return new_args


def _slice_records(records: List, start: int, end: Optional[int], step: int, max_records: Optional[int]) -> List:
    sliced = records[start:end:step]
    if max_records is not None:
        return sliced[:max_records]
    return sliced


def _compute_ranks(records: List, predictions: Dict[str, Dict[str, float]]) -> List[int]:
    ranks: List[int] = []
    for record in records:
        prediction = predictions.get(record.uid, {})
        ordered = sorted(prediction.items(), key=lambda item: item[1], reverse=True)
        rank: Optional[int] = None
        for position, (candidate, _) in enumerate(ordered, start=1):
            if candidate == record.name:
                rank = position
                break
        if rank is None:
            raise RuntimeError(
                f"Could not find true label '{record.name}' in prediction candidates for uid={record.uid}"
            )
        ranks.append(rank)
    return ranks


def _compute_mrr(ranks: List[int]) -> float:
    if not ranks:
        return 0.0
    total = 0.0
    n = float(len(ranks))
    for rank in ranks:
        total += 1.0 / rank
    return total / n


def _compute_acc(ranks: List[int]) -> float:
    if not ranks:
        return 0.0
    hits = 0
    for rank in ranks:
        if rank == 1:
            hits += 1
    return hits / float(len(ranks))


def _top_predictions(records: List, predictions: Dict[str, Dict[str, float]], n: int) -> List[List[str]]:
    if n <= 0:
        return []
    out: List[List[str]] = []
    for record in records:
        prediction = predictions.get(record.uid, {})
        ordered = sorted(prediction.items(), key=lambda item: item[1], reverse=True)
        out.append([name for name, _ in ordered[:n]])
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pure TRI rank evaluation")
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--pipeline_in', type=str, default=None)
    parser.add_argument('--split', type=str, default=None)
    parser.add_argument('--task_id', type=int, default=None)
    args = parser.parse_args()

    task_id = resolve_task_id(args.task_id)
    config_path = apply_task_template(args.config, task_id)
    args = normalize_config(_read_yaml(config_path), args)

    for key in ('data_in', 'split', 'result_in', 'pipeline_in', 'risk_in', 'save_to_jsonl'):
        setattr(args, key, apply_task_template(getattr(args, key, None), task_id))

    if not args.data or not args.data_in:
        raise ValueError("data and data_in are required")
    if not args.pipeline_in:
        raise ValueError("pipeline_in is required for TRI rank evaluation")

    original_records = list(get_adapter(args.data, data=args.data, data_in=args.data_in, split=args.split).iter_records())

    if args.result_in is not None:
        records, _ = build_dataset_from_results(args.result_in, original_records)
    else:
        records = original_records

    records = _slice_records(records, args.start, args.end, args.step, args.max_records)

    detector = TRIDetector(dataset_name=args.data)
    detector.load(args.pipeline_in)
    predictions = detector.predict(records)

    unknown_names = sorted({record.name for record in records if record.name not in detector.name_to_label})
    if unknown_names:
        sample = unknown_names[:10]
        raise ValueError(
            f"Found {len(unknown_names)} record names not present in checkpoint label mapping. Sample: {sample}. This usually indicates a split/checkpoint mismatch."
        )

    ranks = _compute_ranks(records, predictions)
    mrr = _compute_mrr(ranks)
    acc = _compute_acc(ranks)
    first_preds = _top_predictions(records, predictions, int(args.n_first_predictions or 0))

    writer = None
    if args.save_to_jsonl:
        writer = open(args.save_to_jsonl, 'w', encoding='utf-8')

    if writer is None:
        print(f"Mean Reciprocal Rank (MRR): {mrr:.6f}")
        print(f"Accuracy (ACC): {acc:.6f}")

    for i, record in enumerate(records):
        row = {
            "uid": record.uid,
            "name": record.name,
            "rank": ranks[i],
        }
        if args.n_first_predictions > 0:
            row["top_predictions"] = first_preds[i]

        if writer is None:
            print(f"Record UID: {record.uid} | Evaluated Rank: {ranks[i]}")
            if args.n_first_predictions > 0:
                print(f"Top {args.n_first_predictions} Predictions for UID {record.uid}: {first_preds[i]}")
            if args.full_record:
                print(record)
        else:
            writer.write(json.dumps(row) + '\n')

    if writer is not None:
        writer.close()
