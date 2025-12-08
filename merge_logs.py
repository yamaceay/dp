#!/usr/bin/env python3
"""Merge tab-format metric logs into a JSONL summary."""

from __future__ import annotations

import argparse
import ast
import json
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

KEY_RENAMES = {
    "divergence": {"mean_divergence": "mean_bertscore"},
    "utility": {
        "f1_predict_country": "f1_country",
        "f1_predict_year": "f1_year",
    },
}

PARAM_SUFFIXES: List[Tuple[str, str]] = [
    ("_risk_tolerance", "rho"),
    ("_pii_confidence", "lambda"),
    ("_epsilon", "epsilon"),
    ("_lambda", "lambda"),
    ("_rho", "rho"),
    ("_k", "k"),
]


@dataclass
class LogSpec:
    prefix: str
    path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge tab-style logs (e.g., privacy/divergence/utility) into JSONL.",
    )
    parser.add_argument(
        "--log",
        dest="logs",
        action="append",
        metavar="PREFIX:PATH",
        help="A tab log to merge, specified as prefix:path (e.g., divergence:logs/reddit_divergence.jsonl).",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("logs/tab_results.jsonl"),
        help="Output JSONL file.",
    )
    return parser.parse_args()


def parse_log_specs(raw_specs: Iterable[str]) -> List[LogSpec]:
    specs: List[LogSpec] = []
    for raw in raw_specs:
        if ":" not in raw:
            raise ValueError(f"Log specification '{raw}' must be PREFIX:PATH")
        prefix, path_text = raw.split(":", 1)
        specs.append(LogSpec(prefix=prefix.strip(), path=Path(path_text.strip())))
    return specs


def parse_payload(raw: str) -> Dict[str, float]:
    payload = raw.strip().rstrip(",")
    candidate = f"{{{payload}}}"
    try:
        parsed = ast.literal_eval(candidate)
    except (SyntaxError, ValueError) as exc:
        raise ValueError(f"Unable to parse payload: {raw}") from exc
    if not isinstance(parsed, dict):
        raise ValueError(f"Payload must be a dict literal: {raw}")
    return parsed


def read_tab_file(path: Path) -> Dict[str, Dict[str, float]]:
    rows: Dict[str, Dict[str, float]] = OrderedDict()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            if "|" not in stripped:
                raise ValueError(f"Line missing '|' separator in {path}: {stripped}")
            label_part, payload_part = stripped.split("|", 1)
            label = label_part.strip()
            payload = parse_payload(payload_part)
            rows[label] = payload
    return rows


def normalize_label(label: str) -> str:
    if "?" in label:
        return label
    if "-" not in label:
        return label
    prefix, value = label.rsplit("-", 1)
    for suffix, alias in PARAM_SUFFIXES:
        if prefix.endswith(suffix):
            base = prefix[: -len(suffix)].rstrip("_")
            if not base:
                break
            return f"{base}?{alias}={value}"
    return label


def merge_logs(specs: List[LogSpec]) -> List[Dict[str, float]]:
    results: Dict[str, Dict[str, float]] = OrderedDict()
    for spec in specs:
        data = read_tab_file(spec.path)
        rename_map = KEY_RENAMES.get(spec.prefix, {})
        for label, metrics in data.items():
            entry = results.setdefault(label, {})
            for key, value in metrics.items():
                metric_key = rename_map.get(key, key)
                namespaced = f"{spec.prefix}_{metric_key}"
                entry[namespaced] = value
    merged_rows = []
    for label, metrics in results.items():
        row = {"name": normalize_label(label)}
        row.update(metrics)
        merged_rows.append(row)
    return merged_rows


def write_jsonl(rows: List[Dict[str, float]], path: Path) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def main() -> None:
    args = parse_args()
    if not args.logs:
        raise SystemExit("At least one --log PREFIX:PATH argument is required.")
    specs = parse_log_specs(args.logs)
    rows = merge_logs(specs)
    write_jsonl(rows, args.output)


if __name__ == "__main__":
    main()
