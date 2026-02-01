from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Protocol, Tuple
import json
from tqdm import tqdm

from dp.loaders.base import DatasetAdapter, DatasetRecord, TextAnnotation

class RewriterProtocol(Protocol):
    rewriting_pipeline: Any
    def rewrite(self, text: str, **kwargs) -> str: ...

@dataclass
class AttackerDatasetRecord:
    name: str
    train_texts: List[str] = field(default_factory=list)
    eval_texts: List[str] = field(default_factory=list)

class AttackerDatasetAdapter:
    def __init__(
        self,
        adapter: DatasetAdapter,
        **kwargs,
    ) -> None:
        self.adapter = adapter
        self._cache_map: Optional[Dict[str, Dict[str, Any]]] = None
        self._starting_anonymizations_by_idx: Optional[List[List[TextAnnotation]]] = None
        self._starting_replacement: Optional[str] = None

    def set_rewriter(self, rewriter: RewriterProtocol) -> None:
        self.rewriter = rewriter

    def set_starting_anonymizations(
        self,
        annotations_by_idx: Optional[List[List[TextAnnotation]]],
        replacement: Optional[str] = None,
    ) -> None:
        if annotations_by_idx is None:
            self._starting_anonymizations_by_idx = None
        else:
            self._starting_anonymizations_by_idx = list(annotations_by_idx)
        if replacement is not None:
            if not isinstance(replacement, str) or not replacement:
                raise ValueError("replacement must be a non-empty str when provided")
        self._starting_replacement = replacement

    def set_cache(
        self,
        cache: Optional[List[AttackerDatasetRecord]] = None,
        cache_map: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> None:
        if cache_map is not None:
            self._cache_map = cache_map
            return
        if cache is not None:
            self._cache_map = {
                r.name: {
                    "train_texts": r.train_texts,
                    "eval_texts": r.eval_texts,
                }
                for r in cache
            }
            return
        self._cache_map = None

    def load_cache_from_jsonl(self, path: str) -> None:
        mapping = load_attacker_extensions_jsonl(path)
        self.set_cache(cache_map=mapping)

def merge_records(grouped_train_texts: Dict[str, List[str]], grouped_eval_texts: Dict[str, List[str]]) -> Iterable[DatasetRecord]:
    common_names = set(grouped_train_texts.keys()).intersection(set(grouped_eval_texts.keys()))
    attacker_records: List[AttackerDatasetRecord] = []
    for name in common_names:
        train_texts = grouped_train_texts.get(name, [])
        eval_texts = grouped_eval_texts.get(name, [])
        attacker_records.append(
            AttackerDatasetRecord(
                name=name,
                train_texts=train_texts,
                eval_texts=eval_texts,
            )
        )
    return attacker_records

def save_attacker_extensions_jsonl(path: str, records: Iterable[AttackerDatasetRecord]) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for r in records:
            obj = {
                "name": r.name,
                "train_texts": list(r.train_texts or []),
                "eval_texts": list(r.eval_texts or []),
            }
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def load_attacker_extensions_jsonl(path: str) -> Dict[str, Dict[str, Any]]:
    mapping: Dict[str, Dict[str, Any]] = {}
    in_path = Path(path)
    if not in_path.exists():
        raise FileNotFoundError(f"Attacker extensions file not found: {path}")
    with in_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            name = str(obj.get("name"))
            mapping[name] = {
                "train_texts": _normalize_texts(obj.get("train_texts")),
                "eval_texts": _normalize_texts(obj.get("eval_texts")),
            }
    return mapping

def _normalize_texts(raw: Any) -> List[str]:
    if raw is None:
        return []
    if isinstance(raw, str):
        return [raw] if raw else []
    if isinstance(raw, list):
        out: List[str] = []
        for item in raw:
            if not isinstance(item, str) or not item:
                raise ValueError("texts entries must be non-empty strings")
            out.append(item)
        return out
    raise ValueError("Unsupported texts value")