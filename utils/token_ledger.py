from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple


@dataclass
class TokenAddition:
    text: str
    start: Optional[int] = None
    end: Optional[int] = None


@dataclass
class TokenEntry:
    index: int
    original_text: str
    start: int
    end: int
    text: str
    deleted: bool = False
    additions: List[TokenAddition] = field(default_factory=list)

    @property
    def span(self) -> Tuple[int, int]:
        return self.start, self.end


class TokenLedger:
    def __init__(self, source_text: str, spans: Sequence[Tuple[int, int]]) -> None:
        self._entries: List[TokenEntry] = [
            TokenEntry(
                index=idx,
                original_text=source_text[spans[idx][0]:spans[idx][1]],
                start=spans[idx][0],
                end=spans[idx][1],
                text=source_text[spans[idx][0]:spans[idx][1]],
            )
            for idx in range(len(spans))
        ]

    def __len__(self) -> int:
        return len(self._entries)

    def entry(self, index: int) -> TokenEntry:
        return self._entries[index]

    def replace(self, index: int, text: str) -> None:
        entry = self._entries[index]
        if entry.deleted:
            return
        entry.text = text

    def delete(self, index: int) -> None:
        entry = self._entries[index]
        entry.deleted = True
        entry.text = ""
        entry.additions.clear()

    def add_after(self, index: int, text: str) -> None:
        entry = self._entries[index]
        insertion_point = entry.end
        entry.additions.append(TokenAddition(text=text, start=insertion_point, end=insertion_point))

    def iter_tokens(self) -> Iterator[str]:
        for entry in self._entries:
            if entry.deleted:
                continue
            yield entry.text
            for addition in entry.additions:
                yield addition.text

    def iter_entries(self) -> Iterator[TokenEntry]:
        return iter(self._entries)

    def build_context(self) -> List[str]:
        return list(self.iter_tokens())

    def render(self, detokenize: Callable[[List[str]], str]) -> str:
        return detokenize(self.build_context())

    def edits_metadata(self) -> List[Dict[str, object]]:
        metadata: List[Dict[str, object]] = []
        for entry in self._entries:
            if entry.deleted:
                metadata.append(
                    {
                        "span": entry.span,
                        "text": entry.original_text,
                        "kind": "deleted",
                    }
                )
                continue
            if entry.text != entry.original_text:
                metadata.append(
                    {
                        "span": entry.span,
                        "text": entry.text,
                        "kind": "replaced",
                    }
                )
            for addition in entry.additions:
                metadata.append(
                    {
                        "span": (addition.start, addition.end) if addition.start is not None else None,
                        "text": addition.text,
                        "kind": "added",
                    }
                )
        return metadata

    def surviving_spans(self) -> Iterable[Tuple[int, int]]:
        for entry in self._entries:
            if entry.deleted:
                continue
            yield entry.span
