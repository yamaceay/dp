from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

from dp.utils.token_edits import apply_token_edits

@dataclass
class TokenAddition:
    text: str
    start: Optional[int] = None
    end: Optional[int] = None
    source: Optional[str] = None


@dataclass
class TokenEntry:
    index: int
    original_text: str
    start: int
    end: int
    text: str
    deleted: bool = False
    additions: List[TokenAddition] = field(default_factory=list)
    edit_source: Optional[str] = None

    @property
    def span(self) -> Tuple[int, int]:
        return self.start, self.end


@dataclass
class GapEntry:
    start: int
    end: int
    text: str


class TokenLedger:
    def __init__(self, source_text: str, spans: Sequence[Tuple[int, int]]) -> None:
        self._source_text = source_text
        self._entries: List[TokenEntry] = []
        self._gaps: List[GapEntry] = []
        
        sorted_spans = sorted(enumerate(spans), key=lambda x: x[1][0])
        
        prev_end = 0
        for original_idx, (start, end) in sorted_spans:
            if start > prev_end:
                self._gaps.append(GapEntry(start=prev_end, end=start, text=source_text[prev_end:start]))
            
            self._entries.append(
                TokenEntry(
                    index=original_idx,
                    original_text=source_text[start:end],
                    start=start,
                    end=end,
                    text=source_text[start:end],
                )
            )
            prev_end = end
        
        if prev_end < len(source_text):
            self._gaps.append(GapEntry(start=prev_end, end=len(source_text), text=source_text[prev_end:]))

        self._active_edit_source: Optional[str] = None
        self._emit_edit_sources: bool = False

    @property
    def active_edit_source(self) -> Optional[str]:
        return self._active_edit_source

    def set_active_edit_source(self, source: Optional[str]) -> None:
        self._active_edit_source = source

    def set_emit_edit_sources(self, enabled: bool) -> None:
        self._emit_edit_sources = bool(enabled)

    def __len__(self) -> int:
        return len(self._entries)

    def entry(self, index: int) -> TokenEntry:
        return self._entries[index]

    def replace(self, index: int, text: str) -> None:
        entry = self._entries[index]
        if entry.deleted:
            return
        entry.text = text
        entry.edit_source = self._active_edit_source

    def delete(self, index: int) -> None:
        entry = self._entries[index]
        entry.deleted = True
        entry.text = ""
        entry.additions.clear()
        entry.edit_source = self._active_edit_source

    def add_after(self, index: int, text: str) -> None:
        entry = self._entries[index]
        insertion_point = entry.end
        entry.additions.append(
            TokenAddition(
                text=text,
                start=insertion_point,
                end=insertion_point,
                source=self._active_edit_source,
            )
        )

    def iter_tokens(self) -> Iterator[str]:
        sorted_entries = sorted(self._entries, key=lambda e: e.start)
        gap_idx = 0
        
        for entry in sorted_entries:
            if gap_idx < len(self._gaps) and self._gaps[gap_idx].start < entry.start:
                yield self._gaps[gap_idx].text
                gap_idx += 1
            
            if entry.deleted:
                continue
            yield entry.text
            for addition in entry.additions:
                yield addition.text
        
        if gap_idx < len(self._gaps):
            yield self._gaps[gap_idx].text

    def iter_entries(self) -> Iterator[TokenEntry]:
        return iter(sorted(self._entries, key=lambda e: e.start))

    def build_context(self) -> List[str]:
        return list(self.iter_tokens())

    def render(self, detokenize: Callable[[List[str]], str]) -> str:
        return detokenize(self.build_context())

    def render_offsets(self, original_text: str) -> str:
        return apply_token_edits(original_text, self.edits_metadata())

    def edits_metadata(self) -> List[Dict[str, object]]:
        metadata: List[Dict[str, object]] = []
        for entry in self._entries:
            if entry.deleted:
                item: Dict[str, object] = {"span": entry.span, "text": entry.original_text, "kind": "deleted"}
                if self._emit_edit_sources and entry.edit_source is not None:
                    item["source"] = entry.edit_source
                metadata.append(item)
                continue
            if entry.text != entry.original_text:
                item = {"span": entry.span, "text": entry.text, "kind": "replaced"}
                if self._emit_edit_sources and entry.edit_source is not None:
                    item["source"] = entry.edit_source
                metadata.append(item)
            for addition in entry.additions:
                item = {
                    "span": (addition.start, addition.end) if addition.start is not None else None,
                    "text": addition.text,
                    "kind": "added",
                }
                if self._emit_edit_sources and addition.source is not None:
                    item["source"] = addition.source
                metadata.append(item)
        return metadata

    def surviving_spans(self) -> Iterable[Tuple[int, int]]:
        for entry in self._entries:
            if entry.deleted:
                continue
            yield entry.span
