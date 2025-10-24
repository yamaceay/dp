from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple, Optional, Callable

from dp.loaders import DatasetRecord

OutputCallback = Callable[[str], None]

def build_output_sink(output_file: Optional[str]) -> OutputCallback:
    if not output_file:
        return print
    path = Path(output_file).expanduser()
    def sink(content: str) -> None:
        normalized = content if content.endswith("\n") else f"{content}\n"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(normalized, encoding="utf-8")
        print(f"Report written to {path}")
    return sink

def uniquify_reddit_records(records: List[DatasetRecord]) -> List[DatasetRecord]:
    counts: Dict[str, int] = {}
    unique_records: List[DatasetRecord] = []
    for index, record in enumerate(records, start=1):
        original_uid = record.uid or ""
        base_uid = original_uid if original_uid else f"record_{index}"
        occurrence = counts.get(base_uid, 0)
        counts[base_uid] = occurrence + 1
        unique_uid = base_uid if occurrence == 0 else f"{base_uid}#{occurrence}"
        metadata = dict(record.metadata)
        persona_uid = metadata.get("persona_uid", original_uid)
        if not persona_uid:
            persona_uid = base_uid
        metadata["persona_uid"] = persona_uid
        metadata["record_index"] = index
        spans = list(record.spans) if record.spans else None
        unique_records.append(
            DatasetRecord(
                text=record.text,
                uid=unique_uid,
                name=record.name,
                spans=spans,
                metadata=metadata,
            )
        )
    return unique_records

def collect_jsonl_sources(*paths: str) -> Dict[str, Path]:
    resolved_paths = [Path(p) for p in paths]
    entries: List[Tuple[str, Path]] = []
    counts: Dict[str, int] = {}
    seen: set[Path] = set()

    def register(path: Path) -> None:
        if path in seen:
            return
        seen.add(path)
        dataset_name = path.parent.parent.stem
        method_name = path.parent.stem
        base = f"{method_name}_{dataset_name}"
        occurrence = counts.get(base, 0)
        counts[base] = occurrence + 1
        name = f"{base}_{occurrence}"
        entries.append((name, path))

    for path in resolved_paths:
        if path.is_file():
            register(path)
        elif path.is_dir():
            for file_path in sorted(path.rglob("*.jsonl")):
                register(file_path)

    return {name: entry_path for name, entry_path in entries}
