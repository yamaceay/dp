#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from dp.loaders import DatasetRecord, get_adapter
from dp.loaders.results import ResultRecord, load_result_records


@dataclass(frozen=True)
class OffsetFitStats:
    total: int
    both: int
    original_only: int
    result_only: int
    neither: int


def resolve_latest(pattern: str) -> str:
    matches = sorted(glob.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"No files match pattern: {pattern}")
    return matches[-1]


def load_original_records(dataset: str, data_in: str, split: Optional[str]) -> List[DatasetRecord]:
    return list(get_adapter(dataset, data_in=data_in, split=split).iter_records())


def build_uid_to_original_text(records: Sequence[DatasetRecord]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for record in records:
        out[str(record.uid)] = record.text or ""
    return out


def build_uid_to_result_entry(
    result_records: Sequence[ResultRecord],
    original_records: Sequence[DatasetRecord],
) -> Dict[str, ResultRecord]:
    uid_to_entry: Dict[str, ResultRecord] = {}
    for pos, entry in enumerate(result_records):
        idx = entry.idx if entry.idx is not None else pos
        if idx < 0 or idx >= len(original_records):
            continue
        uid = str(original_records[idx].uid)
        uid_to_entry[uid] = entry
    return uid_to_entry


def count_token_edits_presence(result_records: Sequence[ResultRecord]) -> Tuple[int, int]:
    with_edits = 0
    for record in result_records:
        if record.annotations.token_edits:
            with_edits += 1
    return with_edits, len(result_records)


def offsets_fit(offsets: Iterable[Sequence[int]], text_len: int) -> bool:
    for pair in offsets:
        if len(pair) < 2:
            return False
        start = int(pair[0])
        end = int(pair[1])
        if start < 0 or end < start or end > text_len:
            return False
    return True


def analyze_risk_fit(
    risk_entries: Sequence[dict],
    uid_to_original: Dict[str, str],
    uid_to_result: Dict[str, ResultRecord],
    max_examples: int,
) -> Tuple[OffsetFitStats, Dict[str, List[str]]]:
    both = 0
    original_only = 0
    result_only = 0
    neither = 0
    total = 0

    examples: Dict[str, List[str]] = {
        "original_only": [],
        "result_only": [],
        "neither": [],
    }

    for entry in risk_entries:
        uid = str(entry.get("uid", ""))
        offsets = entry.get("offsets", [])
        if uid not in uid_to_original or uid not in uid_to_result:
            continue
        total += 1

        original_text = uid_to_original[uid]
        result_text = uid_to_result[uid].text
        fit_original = offsets_fit(offsets, len(original_text))
        fit_result = offsets_fit(offsets, len(result_text))

        if fit_original and fit_result:
            both += 1
        elif fit_original:
            original_only += 1
            if len(examples["original_only"]) < max_examples:
                examples["original_only"].append(uid)
        elif fit_result:
            result_only += 1
            if len(examples["result_only"]) < max_examples:
                examples["result_only"].append(uid)
        else:
            neither += 1
            if len(examples["neither"]) < max_examples:
                examples["neither"].append(uid)

    return OffsetFitStats(
        total=total,
        both=both,
        original_only=original_only,
        result_only=result_only,
        neither=neither,
    ), examples


def load_risk_entries(path: str) -> List[dict]:
    entries: List[dict] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            if isinstance(payload, dict):
                entries.append(payload)
    return entries


def print_ratio(name: str, num: int, den: int) -> None:
    ratio = 0.0 if den == 0 else num / den
    print(f"- {name}: {num}/{den} ({ratio:.2%})")


def main() -> None:
    parser = argparse.ArgumentParser(description="One-off diagnostic for risk offset alignment.")
    parser.add_argument("--dataset", type=str, default="db_bio")
    parser.add_argument("--data-in", type=str, default="data/db_bio")
    parser.add_argument("--split", type=str, default=None)
    parser.add_argument("--risk-in", type=str, default="data/db_bio/tri_risk/shap.jsonl")
    parser.add_argument(
        "--presidio-result",
        type=str,
        default="/netscratch/yay/_ext_outputs/db_bio/presidio/*_db_bio_presidio.jsonl",
        help="File path or glob pattern",
    )
    parser.add_argument(
        "--dpmlm-result",
        type=str,
        default="/netscratch/yay/_ext_outputs/db_bio/dpmlm/*_db_bio_dpmlm_*.jsonl",
        help="File path or glob pattern",
    )
    parser.add_argument("--examples", type=int, default=5)
    args = parser.parse_args()

    original_records = load_original_records(args.dataset, args.data_in, args.split)
    uid_to_original = build_uid_to_original_text(original_records)

    presidio_path = resolve_latest(args.presidio_result)
    dpmlm_path = resolve_latest(args.dpmlm_result)

    presidio_records = load_result_records(presidio_path)
    dpmlm_records = load_result_records(dpmlm_path)

    presidio_uid_map = build_uid_to_result_entry(presidio_records, original_records)
    dpmlm_uid_map = build_uid_to_result_entry(dpmlm_records, original_records)

    risk_entries = load_risk_entries(args.risk_in)

    pres_edits, pres_total = count_token_edits_presence(presidio_records)
    dp_edits, dp_total = count_token_edits_presence(dpmlm_records)

    print("=== Files ===")
    print(f"- presidio_result: {presidio_path}")
    print(f"- dpmlm_result: {dpmlm_path}")
    print(f"- risk_in: {args.risk_in}")

    print("\n=== token_edits coverage ===")
    print_ratio("presidio with token_edits", pres_edits, pres_total)
    print_ratio("dpmlm with token_edits", dp_edits, dp_total)

    pres_stats, pres_examples = analyze_risk_fit(
        risk_entries=risk_entries,
        uid_to_original=uid_to_original,
        uid_to_result=presidio_uid_map,
        max_examples=args.examples,
    )
    dp_stats, dp_examples = analyze_risk_fit(
        risk_entries=risk_entries,
        uid_to_original=uid_to_original,
        uid_to_result=dpmlm_uid_map,
        max_examples=args.examples,
    )

    print("\n=== Risk offset fit vs presidio result file ===")
    print_ratio("fit both original+result", pres_stats.both, pres_stats.total)
    print_ratio("fit original only", pres_stats.original_only, pres_stats.total)
    print_ratio("fit result only", pres_stats.result_only, pres_stats.total)
    print_ratio("fit neither", pres_stats.neither, pres_stats.total)
    if pres_examples["result_only"]:
        print(f"- sample result_only UIDs: {pres_examples['result_only']}")
    if pres_examples["original_only"]:
        print(f"- sample original_only UIDs: {pres_examples['original_only']}")
    if pres_examples["neither"]:
        print(f"- sample neither UIDs: {pres_examples['neither']}")

    print("\n=== Risk offset fit vs dpmlm result file ===")
    print_ratio("fit both original+result", dp_stats.both, dp_stats.total)
    print_ratio("fit original only", dp_stats.original_only, dp_stats.total)
    print_ratio("fit result only", dp_stats.result_only, dp_stats.total)
    print_ratio("fit neither", dp_stats.neither, dp_stats.total)
    if dp_examples["result_only"]:
        print(f"- sample result_only UIDs: {dp_examples['result_only']}")
    if dp_examples["original_only"]:
        print(f"- sample original_only UIDs: {dp_examples['original_only']}")
    if dp_examples["neither"]:
        print(f"- sample neither UIDs: {dp_examples['neither']}")


if __name__ == "__main__":
    main()
