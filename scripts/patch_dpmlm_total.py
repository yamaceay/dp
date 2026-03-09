from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                payload = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path} at line {line_number}") from exc
            if not isinstance(payload, dict):
                raise ValueError(f"Non-object JSON in {path} at line {line_number}")
            rows.append(payload)
    return rows


def _extract_totals(rows: List[Dict[str, Any]], path: Path) -> List[int]:
    totals: List[int] = []
    for index, row in enumerate(rows):
        metadata = row.get("metadata")
        if not isinstance(metadata, dict):
            raise ValueError(f"Missing metadata object in {path} at record index {index}")
        total_value = metadata.get("total")
        try:
            total = int(total_value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Missing/invalid metadata.total in {path} at record index {index}") from exc
        totals.append(total)
    if not totals:
        raise ValueError(f"No records found in {path}")
    return totals


def _patch_rows(rows: List[Dict[str, Any]], totals: List[int], path: Path) -> Tuple[int, int]:
    if len(rows) != len(totals):
        raise ValueError(
            f"Record count mismatch in {path}: expected {len(totals)} rows from reference, found {len(rows)}"
        )

    touched = 0
    for index, row in enumerate(rows):
        metadata = row.get("metadata")
        if metadata is None:
            metadata = {}
            row["metadata"] = metadata
        if not isinstance(metadata, dict):
            raise ValueError(f"Invalid metadata type in {path} at record index {index}")

        previous = metadata.get("total")
        updated = int(totals[index])
        metadata["total"] = updated
        if previous != updated:
            touched += 1

    return touched, len(rows)


def _write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _discover_targets(dataset_root: Path, reference_path: Path) -> List[Path]:
    targets = sorted(p for p in dataset_root.glob("*.jsonl") if p.is_file())
    if not targets:
        raise ValueError(f"No jsonl files found under {dataset_root}")
    if reference_path not in targets:
        targets.append(reference_path)
        targets.sort()
    return targets


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Patch dpmlm metadata.total values in JSONL outputs")
    parser.add_argument("--dataset-root", type=Path, required=True, help="Directory containing dpmlm JSONL outputs")
    parser.add_argument("--reference", type=Path, required=True, help="Reference JSONL file with correct totals by line order")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root.resolve()
    reference_path = args.reference.resolve()

    if not dataset_root.is_dir():
        raise ValueError(f"dataset-root is not a directory: {dataset_root}")
    if not reference_path.is_file():
        raise ValueError(f"reference is not a file: {reference_path}")

    reference_rows = _load_jsonl(reference_path)
    reference_totals = _extract_totals(reference_rows, reference_path)
    targets = _discover_targets(dataset_root, reference_path)

    files_updated = 0
    records_updated = 0
    records_seen = 0

    for path in targets:
        rows = _load_jsonl(path)
        touched, seen = _patch_rows(rows, reference_totals, path)
        _write_jsonl(path, rows)
        files_updated += 1
        records_updated += touched
        records_seen += seen

    print(
        json.dumps(
            {
                "dataset_root": str(dataset_root),
                "reference": str(reference_path),
                "files_updated": files_updated,
                "records_seen": records_seen,
                "records_updated": records_updated,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
