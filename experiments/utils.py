from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple, Optional, Callable, Sequence, Any

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

def _first_text(*values: Any) -> str:
    for value in values:
        if isinstance(value, str):
            text = value.strip()
            if text:
                return text
        elif isinstance(value, (list, tuple, set)):
            nested = _first_text(*value)
            if nested:
                return nested
        elif value is not None:
            text = str(value).strip()
            if text:
                return text
    return ""

def uniquify_records(
    records: Sequence[DatasetRecord],
    *,
    identity_fields: Sequence[str] = ("uid", "name"),
    metadata_identity_keys: Sequence[str] = ("persona_uid", "uid", "name"),
    fallback_prefix: str = "record",
) -> List[DatasetRecord]:
    counts: Dict[str, int] = {}
    output: List[DatasetRecord] = []
    for index, record in enumerate(records, start=1):
        identity_sources: List[Any] = []
        for field in identity_fields:
            identity_sources.append(getattr(record, field, None))
        metadata = dict(record.metadata or {})
        for key in metadata_identity_keys:
            identity_sources.append(metadata.get(key))
        base = _first_text(*identity_sources)
        if not base:
            base = f"{fallback_prefix}_{index}"
        occurrence = counts.get(base, 0)
        counts[base] = occurrence + 1
        uid = base if occurrence == 0 else f"{base}#{occurrence}"
        persona_uid = _first_text(metadata.get("persona_uid"), base)
        metadata["persona_uid"] = persona_uid
        metadata["record_index"] = index
        spans = list(record.spans) if record.spans else None
        output.append(
            DatasetRecord(
                text=record.text,
                uid=uid,
                name=record.name,
                spans=spans,
                metadata=metadata,
            )
        )
    return output

def uniquify_reddit_records(records: Sequence[DatasetRecord]) -> List[DatasetRecord]:
    return uniquify_records(
        records,
        identity_fields=("uid", "name"),
        metadata_identity_keys=("persona_uid", "uid", "name"),
        fallback_prefix="record",
    )

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
