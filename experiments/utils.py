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
