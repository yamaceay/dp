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
class AttackerDatasetRecord(DatasetRecord):
    background_knowledge: List[Tuple[str, str]] = field(default_factory=list)
    rewrited_text: Optional[str] = None

class AttackerDatasetAdapter:
    def __init__(
        self,
        adapter: DatasetAdapter,
        max_background_tokens: int = 512,
        rewriter_max_length: int = 150,
        rewriter_min_length: int = 40,
    ) -> None:
        self.adapter = adapter
        self.max_background_tokens = max_background_tokens
        self.rewriter_max_length = rewriter_max_length
        self.rewriter_min_length = rewriter_min_length
        self.rewriter: Optional[RewriterProtocol] = None
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
                r.uid: {
                    "background_knowledge": r.background_knowledge,
                    "rewrited_text": r.rewrited_text,
                }
                for r in cache
            }
            return
        self._cache_map = None

    def load_cache_from_jsonl(self, path: str) -> None:
        mapping = load_attacker_extensions_jsonl(path)
        self.set_cache(cache_map=mapping)

    def extract_background_knowledge(self, record: DatasetRecord) -> List[Tuple[str, str]]:
        raise NotImplementedError

    def clean_metadata(self, record: DatasetRecord) -> Dict[str, Any]:
        return record.metadata

    def rewrite_original_text(self, record: DatasetRecord) -> str:
        if not self.rewriter:
            raise RuntimeError("rewriter is not set; call set_rewriter() first or load cache with summaries")
        return self.rewriter.rewrite(
            text=record.text,
            max_length=self.rewriter_max_length,
            min_length=self.rewriter_min_length,
            do_sample=False,
        )

    def _apply_starting_anonymizations(self, text: str, annotations: List[TextAnnotation]) -> str:
        if not annotations:
            return text
        masked = text
        for ann in sorted(annotations, key=lambda a: a.start, reverse=True):
            if not isinstance(ann.start, int) or not isinstance(ann.end, int):
                raise ValueError("Starting anonymization annotation start and end must be integers")
            if ann.start < 0 or ann.end < 0 or ann.start >= ann.end:
                raise ValueError("Starting anonymization annotation has invalid start/end span")
            if ann.end > len(masked):
                raise ValueError("Starting anonymization annotation end exceeds text length")
            repl: Optional[str] = None
            if isinstance(ann.replacement, str) and ann.replacement:
                repl = ann.replacement
            elif isinstance(ann.label, str) and ann.label:
                repl = f"[{ann.label}]"
            elif isinstance(self._starting_replacement, str) and self._starting_replacement:
                repl = self._starting_replacement
            if repl is None:
                raise ValueError("Starting anonymization span has no replacement and no label; cannot mask")
            masked = masked[: ann.start] + repl + masked[ann.end :]
        return masked

    def iter_records(self, progress: bool = False) -> Iterable[AttackerDatasetRecord]:
        records_list = list(self.adapter.iter_records())
        iterator = iter(records_list)
        if progress:
            iterator = tqdm(records_list, desc="Processing attacker records", total=len(records_list))
        for idx, record in enumerate(iterator):
            record_for_processing = record
            if self._starting_anonymizations_by_idx is not None and idx < len(self._starting_anonymizations_by_idx):
                anns = self._starting_anonymizations_by_idx[idx]
                if anns:
                    record_for_processing = DatasetRecord(
                        text=self._apply_starting_anonymizations(record.text, anns),
                        uid=record.uid,
                        name=record.name,
                        spans=record.spans,
                        metadata=record.metadata,
                    )

            rewr = record_for_processing.text
            if self._cache_map is not None and record.uid in self._cache_map:
                ext = self._cache_map.get(record.uid, {})
                bk = ext.get("background_knowledge", [])
                rewr = ext.get("rewrited_text")
            else:
                bk = self.extract_background_knowledge(record_for_processing)
                if self.rewriter:
                    rewr = self.rewrite_original_text(record_for_processing)
            metadata = self.clean_metadata(record)
            yield AttackerDatasetRecord(
                text=record.text,
                uid=record.uid,
                name=record.name,
                spans=record.spans,
                metadata=metadata,
                background_knowledge=bk,
                rewrited_text=rewr,
            )


def save_attacker_extensions_jsonl(path: str, records: Iterable[AttackerDatasetRecord]) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for r in records:
            obj = {
                "uid": r.uid,
                "background_knowledge": r.background_knowledge,
                "rewrited_text": r.rewrited_text,
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
            uid = str(obj.get("uid"))
            mapping[uid] = {
                "background_knowledge": obj.get("background_knowledge", []),
                "rewrited_text": obj.get("rewrited_text"),
            }
    return mapping