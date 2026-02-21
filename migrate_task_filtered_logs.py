from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, List, Set, Tuple


def parse_task_id_from_log_name(path: Path) -> int | None:
    match = re.search(r"_exp_task_([0-9]+)\.jsonl$", path.name)
    if not match:
        return None
    return int(match.group(1))


def parse_task_id_from_source(source: Any) -> int | None:
    if not isinstance(source, str):
        return None
    match = re.search(r"_task_([0-9]+)", source)
    if not match:
        return None
    return int(match.group(1))


def read_jsonl(path: Path) -> List[dict[str, Any]]:
    rows: List[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as reader:
        for raw in reader:
            line = raw.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: List[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as writer:
        for row in rows:
            writer.write(json.dumps(row, ensure_ascii=False))
            writer.write("\n")


def filter_rows_for_task(rows: List[dict[str, Any]], task_id: int) -> Tuple[List[dict[str, Any]], int]:
    allowed_evaluations: Set[str] = set()
    removed = 0
    for row in rows:
        if row.get("type") != "evaluation":
            continue
        source_task_id = parse_task_id_from_source(row.get("source"))
        if source_task_id == task_id:
            name = row.get("name")
            if isinstance(name, str) and name:
                allowed_evaluations.add(name)

    kept: List[dict[str, Any]] = []
    active_evaluation_kept = True
    for row in rows:
        record_type = row.get("type")
        if record_type == "evaluation":
            name = row.get("name")
            active_evaluation_kept = bool(isinstance(name, str) and name in allowed_evaluations)
            if isinstance(name, str) and name in allowed_evaluations:
                kept.append(row)
            else:
                removed += 1
            continue

        evaluation_name = row.get("evaluation")
        if isinstance(evaluation_name, str):
            if evaluation_name in allowed_evaluations:
                kept.append(row)
            else:
                removed += 1
            continue

        if record_type in {"experiment", "original_rank"}:
            kept.append(row)
            continue

        if not active_evaluation_kept:
            removed += 1
            continue

        kept.append(row)

    return kept, removed


def migrate_logs(logs_dir: Path, apply: bool, backup_suffix: str) -> None:
    paths = sorted(logs_dir.glob("*_exp_task_*.jsonl"))
    print(f"Found {len(paths)} log files to process in {logs_dir}")
    touched = 0
    total_removed = 0
    for path in paths:
        task_id = parse_task_id_from_log_name(path)
        if task_id is None:
            continue
        rows = read_jsonl(path)
        filtered_rows, removed = filter_rows_for_task(rows, task_id)
        if removed <= 0:
            continue
        touched += 1
        total_removed += removed
        print(f"{path}: remove={removed} keep={len(filtered_rows)}")
        if apply:
            backup_path = path.with_name(path.name + backup_suffix)
            if not backup_path.exists():
                backup_path.write_bytes(path.read_bytes())
            write_jsonl(path, filtered_rows)
    print(f"files_touched={touched} removed_rows={total_removed} apply={apply}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Filter task log files to keep only matching source task rows.")
    parser.add_argument("--logs-dir", type=str, default="logs")
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--backup-suffix", type=str, default=".bak")
    args = parser.parse_args()
    migrate_logs(Path(args.logs_dir), args.apply, args.backup_suffix)


if __name__ == "__main__":
    main()
