import json
import sys
from typing import Dict
import numpy as np
import argparse

from dp.loaders import get_adapter
from dp.loaders.results import build_dataset_from_results, load_result_records
from dp.tri.base import TRIDetector
from dp.utils.splitter import TextSplitter
from dp.utils.tasking import apply_task_template, resolve_task_id
from runtime.config_loader import _read_yaml

from dp.utils.token_edits import map_original_offset_to_result, map_result_offset_to_original


def _normalize_offsets(offsets):
    normalized = []
    for span in offsets:
        if not isinstance(span, (list, tuple)) or len(span) < 2:
            raise ValueError(f"Invalid offset span: {span}")
        start = int(span[0])
        end = int(span[1])
        if start < 0 or end < start:
            raise ValueError(f"Invalid offset span: ({start}, {end})")
        normalized.append((start, end))
    return normalized


def _token_span_set(text: str, splitter: TextSplitter):
    return {(s, e) for s, e, _ in splitter.tokenize_with_spans(text)}


def _all_in_set(offsets, span_set):
    return all((s, e) in span_set for s, e in offsets)


def _map_result_to_original_all(offsets, edits):
    mapped = []
    for start, end in offsets:
        mapped_start, mapped_end = map_result_offset_to_original(start, end, edits)
        mapped.append((int(mapped_start), int(mapped_end)))
    return mapped


def _map_original_to_result_all(offsets, edits):
    mapped = []
    for start, end in offsets:
        mapped_span = map_original_offset_to_result(start, end, edits)
        if mapped_span is None:
            raise ValueError(f"Offset ({start}, {end}) overlaps an edited/deleted region")
        mapped_start, mapped_end = mapped_span
        mapped.append((int(mapped_start), int(mapped_end)))
    return mapped


def _align_offsets(
    uid: str,
    offsets,
    target_mode: str,
    original_text: str,
    result_text: str,
    token_edits,
    splitter: TextSplitter,
):
    normalized_offsets = _normalize_offsets(offsets)
    original_spans = _token_span_set(original_text, splitter)
    result_spans = _token_span_set(result_text, splitter)

    fits_original = _all_in_set(normalized_offsets, original_spans)
    fits_result = _all_in_set(normalized_offsets, result_spans)

    source_mode = None
    if fits_original and not fits_result:
        source_mode = "original"
    elif fits_result and not fits_original:
        source_mode = "result"
    elif fits_original and fits_result:
        source_mode = target_mode

    if source_mode is None:
        if not token_edits:
            raise ValueError(
                f"UID={uid}: offsets do not match token spans in either original or result text and no token_edits are available"
            )

        mapped_to_original = _map_result_to_original_all(normalized_offsets, token_edits)
        mapped_to_result = _map_original_to_result_all(normalized_offsets, token_edits)

        mapped_result_is_original = _all_in_set(mapped_to_original, original_spans)
        mapped_original_is_result = _all_in_set(mapped_to_result, result_spans)

        if mapped_result_is_original and not mapped_original_is_result:
            source_mode = "result"
        elif mapped_original_is_result and not mapped_result_is_original:
            source_mode = "original"
        else:
            raise ValueError(
                f"UID={uid}: unable to infer offset coordinate space unambiguously"
            )

    if source_mode == target_mode:
        aligned_offsets = normalized_offsets
    elif source_mode == "result" and target_mode == "original":
        if not token_edits:
            raise ValueError(f"UID={uid}: cannot map result offsets to original without token_edits")
        aligned_offsets = _map_result_to_original_all(normalized_offsets, token_edits)
    elif source_mode == "original" and target_mode == "result":
        if not token_edits:
            raise ValueError(f"UID={uid}: cannot map original offsets to result without token_edits")
        aligned_offsets = _map_original_to_result_all(normalized_offsets, token_edits)
    else:
        raise ValueError(f"UID={uid}: unsupported mode conversion {source_mode} -> {target_mode}")

    target_spans = original_spans if target_mode == "original" else result_spans
    if not _all_in_set(aligned_offsets, target_spans):
        raise ValueError(f"UID={uid}: aligned offsets do not match token spans in target text ({target_mode})")

    return aligned_offsets


def normalize_config(config: Dict, args: argparse.Namespace) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=None, help='Path to config file')
    parser.add_argument('--data', type=str, default=None, help='Dataset name')
    parser.add_argument('--data_in', type=str, default=None, help='Path to input data file')
    parser.add_argument('--split', type=str, default=None, help='Path to split indices file')
    parser.add_argument('--result_in', type=str, default=None, help='Path to anonymization results JSONL file')
    parser.add_argument('--pipeline_in', type=str, default=None, help='Path to anonymization pipeline JSON file')
    parser.add_argument('--risk_in', type=str, default=None, help='Path to anonymization risks JSONL file')
    parser.add_argument('--start', type=int, default=0, help='Start index for records to process')
    parser.add_argument('--end', type=int, default=None, help='End index for records to process')
    parser.add_argument('--step', type=int, default=1, help='Step size for records to process')
    parser.add_argument('--max_records', type=int, default=None, help='Maximum number of records to load')
    parser.add_argument('--full_record', action='store_true', help='Whether to display full record information')
    parser.add_argument('--save_to_jsonl', type=str, default=None, help='Path to save the output records as JSONL')
    parser.add_argument('--offset_mode', type=str, choices=['original', 'result'], default='result', help='Whether risk offsets are in original or result text coordinates')
    parser.add_argument('--abs', action='store_true', help='Use absolute scores')
    parser.add_argument('--n_first_predictions', type=int, default=0, help='Number of top predictions to print per record')

    new_args = parser.parse_args([])
    for key, value in config.items():
        if hasattr(new_args, key):
            setattr(new_args, key, value)
    for key, value in vars(args).items():
        if value is not None:
            setattr(new_args, key, value)
    return new_args

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test RiskAnonymizer")
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--pipeline_in', type=str, default=None, help='Path to anonymization pipeline JSON file')
    parser.add_argument('--split', type=str, default=None, help='Path to split indices file')
    parser.add_argument('--task_id', type=int, default=None, help='Task id for task-aware path templates')
    args = parser.parse_args()
    task_id = resolve_task_id(args.task_id)
    config_path = apply_task_template(args.config, task_id)
    args = normalize_config(_read_yaml(config_path), args)
    for key in ('data_in', 'split', 'result_in', 'pipeline_in', 'risk_in', 'save_to_jsonl'):
        setattr(args, key, apply_task_template(getattr(args, key, None), task_id))

    if not args.data or not args.data_in:
        raise ValueError("data and data_in are required")

    splitter = TextSplitter()

    original_records = list(get_adapter(args.data, 
                                        data_in=args.data_in,
                                        split=args.split).iter_records())
    original_text_by_uid = {r.uid: r.text for r in original_records}
    args.end = args.end if args.end is not None else len(original_records)

    if args.result_in is not None:
        records, indices = build_dataset_from_results(args.result_in, original_records)
        result_records = load_result_records(args.result_in)
        result_text_by_uid = {original_records[rr.idx].uid: rr.text for rr in result_records if rr.idx is not None}
        token_edits_by_uid = {
            original_records[rr.idx].uid: [te.to_dict() for te in rr.annotations.token_edits]
            for rr in result_records
            if rr.idx is not None
        }
    else:
        records = original_records
        result_records = original_records
        result_text_by_uid = {r.uid: r.text for r in result_records}
        token_edits_by_uid = {}

    args.end = min(args.end, len(records))

    if args.risk_in is not None:
        risk_by_uid = {}
        with open(args.risk_in, 'r', encoding='utf-8') as f:
            for line in f:
                entry = json.loads(line.strip())
                risk_by_uid[entry['uid']] = entry

        scores = []
        offsets = []
        tokens = []
        for record in records:
            entry = risk_by_uid.get(record.uid)
            if entry is None:
                raise ValueError(f"No risk entry found for UID={record.uid}")
            original_text = original_text_by_uid.get(record.uid)
            if original_text is None:
                raise ValueError(f"No original text found for UID={record.uid}")
            result_text = result_text_by_uid.get(record.uid)
            if result_text is None:
                raise ValueError(f"No result text found for UID={record.uid}")
            token_edits = token_edits_by_uid.get(record.uid, [])

            offsets_record = _align_offsets(
                uid=str(record.uid),
                offsets=entry['offsets'],
                target_mode=str(args.offset_mode),
                original_text=original_text,
                result_text=result_text,
                token_edits=token_edits,
                splitter=splitter,
            )

            if args.offset_mode == 'result':
                text_for_lookup = result_text
            else:
                text_for_lookup = original_text
            scores_record = entry['scores']

            tokens_record = []
            for s, e in offsets_record:
                tokens_record.append(text_for_lookup[s:e])
            
            if args.abs:
                scores_idx = np.argsort(np.abs(scores_record))[::-1]
                scores_record = [scores_record[i] for i in scores_idx]
                offsets_record = [offsets_record[i] for i in scores_idx]
                tokens_record = [tokens_record[i] for i in scores_idx]

            scores.append(scores_record)
            offsets.append(offsets_record)
            tokens.append(tokens_record)

        scores = scores[args.start:args.end:args.step][:args.max_records or len(scores)]
        offsets = offsets[args.start:args.end:args.step][:args.max_records or len(offsets)]
        tokens = tokens[args.start:args.end:args.step][:args.max_records or len(tokens)]
    else:
        risk_by_uid = {}

    records = records[args.start:args.end:args.step][:args.max_records or len(records)]

    mrr = None
    acc = None
    ranks = None
    first_preds = None

    if args.pipeline_in:
        detector = TRIDetector(dataset_name=args.data)
        detector.load(args.pipeline_in)
        predictions = detector.predict(records)

        unknown_names = sorted({record.name for record in records if record.name not in detector.name_to_label})
        if unknown_names:
            sample = unknown_names[:10]
            raise ValueError(
                f"Found {len(unknown_names)} record names not present in checkpoint label mapping. "
                f"Sample: {sample}. "
                "This usually indicates a split/checkpoint mismatch."
            )

        mrr = 0.0
        acc = 0.0
        ranks = []
        if args.n_first_predictions > 0:
            first_preds = []

        for record in records:
            prediction = predictions.get(record.uid, {})
            ordered = sorted(prediction.items(), key=lambda item: item[1], reverse=True)
            if args.n_first_predictions > 0:
                first_preds.append([x[0] for x in ordered[:args.n_first_predictions]])
            rank = None
            for position, (candidate, _) in enumerate(ordered, start=1):
                if candidate == record.name:
                    rank = position
                    break
            ranks.append(rank)
            if rank is None:
                raise RuntimeError(
                    f"Could not find true label '{record.name}' in prediction candidates for uid={record.uid}. "
                    "This should not happen after label mapping consistency check."
                )
            mrr += 1.0 / (rank * len(records))
            acc += 1.0 / len(records) if rank == 1 else 0.0

    j = 0
    f = None
    if args.save_to_jsonl:
        f = open(args.save_to_jsonl, 'w', encoding='utf-8')
    if not args.save_to_jsonl and args.pipeline_in:
        print(f"Mean Reciprocal Rank (MRR): {mrr:.6f}")
        print(f"Accuracy (ACC): {acc:.6f}")
    for i in range(args.start, args.end, args.step):
        if args.max_records and j >= args.max_records:
            break
        if not args.save_to_jsonl:
            if args.pipeline_in:
                print(f"Record UID: {records[j].uid} | Evaluated Rank: {ranks[j]}")
            if args.full_record and args.risk_in:
                for token, offset, score in zip(tokens[j], offsets[j], scores[j]):
                    print(f"Token: '{token}' | Offset: {offset} | Score: {score}")
            if args.n_first_predictions > 0 and first_preds is not None:
                print(f"Top {args.n_first_predictions} Predictions for UID {records[j].uid}: {first_preds[j]}")
        else:
            if args.pipeline_in:
                f.write(json.dumps({"uid": records[j].uid, "rank": ranks[j]}) + '\n')
            if args.risk_in:
                for token, offset, score in zip(tokens[j], offsets[j], scores[j]):
                    f.write(json.dumps({"uid": records[j].uid, "token": token, "offset": offset, "score": score}) + '\n')
            if args.n_first_predictions > 0 and first_preds is not None:
                f.write(json.dumps({"uid": records[j].uid, "top_predictions": first_preds[j]}) + '\n')
        j += 1
