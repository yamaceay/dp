from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Protocol, Tuple
import json
from tqdm import tqdm

from dp.loaders.base import DatasetAdapter, DatasetRecord

class SummarizerProtocol(Protocol):
    summarization_pipeline: Any
    def summarize(self, text: str, **kwargs) -> str: ...

@dataclass
class AttackerDatasetRecord(DatasetRecord):
    background_knowledge: List[Tuple[str, str]] = field(default_factory=list)
    summarized_text: Optional[str] = None

class AttackerDatasetAdapter:
    def __init__(
        self,
        adapter: DatasetAdapter,
        max_background_tokens: int = 512,
        summarizer_max_length: int = 150,
        summarizer_min_length: int = 40,
    ) -> None:
        self.adapter = adapter
        self.max_background_tokens = max_background_tokens
        self.summarizer_max_length = summarizer_max_length
        self.summarizer_min_length = summarizer_min_length
        self.summarizer: Optional[SummarizerProtocol] = None
        self._cache_map: Optional[Dict[str, Dict[str, Any]]] = None

    def set_summarizer(self, summarizer: SummarizerProtocol) -> None:
        self.summarizer = summarizer

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
                    "summarized_text": r.summarized_text,
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

    def summarize_original_text(self, record: DatasetRecord) -> str:
        if not self.summarizer:
            raise RuntimeError("summarizer is not set; call set_summarizer() first or load cache with summaries")
        return self.summarizer.summarize(
            text=record.text,
            max_length=self.summarizer_max_length,
            min_length=self.summarizer_min_length,
            do_sample=False,
        )

    def iter_records(self, progress: bool = False) -> Iterable[AttackerDatasetRecord]:
        records_list = list(self.adapter.iter_records())
        iterator = iter(records_list)
        if progress:
            iterator = tqdm(records_list, desc="Processing attacker records", total=len(records_list))
        for record in iterator:
            if self._cache_map is not None and record.uid in self._cache_map:
                ext = self._cache_map.get(record.uid, {})
                bk = ext.get("background_knowledge", [])
                summ = ext.get("summarized_text")
            else:
                bk = self.extract_background_knowledge(record)
                summ = self.summarize_original_text(record)
            yield AttackerDatasetRecord(
                text=record.text,
                uid=record.uid,
                name=record.name,
                spans=record.spans,
                metadata=record.metadata,
                background_knowledge=bk,
                summarized_text=summ,
            )


def save_attacker_extensions_jsonl(path: str, records: Iterable[AttackerDatasetRecord]) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for r in records:
            obj = {
                "uid": r.uid,
                "background_knowledge": r.background_knowledge,
                "summarized_text": r.summarized_text,
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
                "summarized_text": obj.get("summarized_text"),
            }
    return mapping