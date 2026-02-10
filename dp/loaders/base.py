"""Base classes and attacker adapter for dataset processing."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional
import itertools


@dataclass(frozen=True)
class TokenEdit:
    kind: str
    span: Optional[tuple[int, int]]
    text: str
    replacement: Optional[str] = None

    @classmethod
    def from_mapping(cls, obj: object) -> "TokenEdit":
        if not isinstance(obj, dict):
            raise TypeError("TokenEdit must be built from a dict")
        kind = obj.get("kind")
        if not isinstance(kind, str) or not kind:
            raise ValueError("TokenEdit.kind must be a non-empty str")
        span_obj = obj.get("span")
        span: Optional[tuple[int, int]] = None
        if span_obj is not None:
            if not isinstance(span_obj, (list, tuple)) or len(span_obj) != 2:
                raise ValueError("TokenEdit.span must be a pair")
            start, end = span_obj
            if not isinstance(start, int) or not isinstance(end, int):
                raise ValueError("TokenEdit.span values must be ints")
            span = (start, end)
        text = obj.get("text", "")
        if not isinstance(text, str):
            raise ValueError("TokenEdit.text must be a str")
        replacement = obj.get("replacement")
        if replacement is not None and not isinstance(replacement, str):
            raise ValueError("TokenEdit.replacement must be a str or null")
        return cls(kind=kind, span=span, text=text, replacement=replacement)

    def to_dict(self) -> Dict[str, object]:
        out: Dict[str, object] = {"kind": self.kind, "text": self.text}
        if self.span is not None:
            out["span"] = [self.span[0], self.span[1]]
        if self.replacement is not None:
            out["replacement"] = self.replacement
        return out


@dataclass
class TextAnnotations:
    spans: list[TextAnnotation] = field(default_factory=list)
    token_edits: list[TokenEdit] = field(default_factory=list)

@dataclass
class TextAnnotation:
    """Representation of a text annotation."""

    start: int
    end: int
    label: Optional[str] = None # only used if category-specific annotations
    text: Optional[str] = None # only used if we want to store the original text span for verbosity
    replacement: Optional[str] = None # only used if we want to store the replacement text for verbosity
    confidence: Optional[float] = None # only used if the annotation is a prediction
    annotator: Optional[str] = None # only used if we want to store who made the annotation
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class DatasetRecord:
    """Normalized representation of a dataset record."""

    text: str
    uid: str = ""
    name: str = ""
    spans: Optional[List[TextAnnotation]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class DatasetAdapter:
    """Base adapter providing a unified interface across datasets."""

    def __init__(self, 
                 data: Optional[str] = None, 
                 data_in: Optional[str] = None, 
                 max_records: Optional[int] = None, 
                 start: Optional[int] = None, 
                 end: Optional[int] = None, 
                 step: Optional[int] = None):
        if data_in is None:
            raise ValueError("data_in must point to a JSONL file")
        data_path = Path(data_in)
        if not data_path.exists():
            raise ValueError(f"data_in path '{data_in}' does not exist or is not a file")
        self.data_in = data_in
        self.max_records = max_records
        self.start = start
        self.end = end
        self.step = step

    def __iter__(self) -> Iterator[DatasetRecord]:
        return iter(self.iter_records())

    def iter_records(self, *args, **kwargs) -> Iterable[DatasetRecord]:
        """Yield normalized dataset records."""
        raise NotImplementedError

    def __len__(self) -> int:
        """Return the size of the dataset if known."""
        raise NotImplementedError

    def _slice_records(self, iterable: Iterable[Any]) -> Iterable[Any]:
        """Apply slicing (start/end/step) before enforcing max_records."""
        return slice_records(iterable, self.start, self.end, self.step, self.max_records)

def slice_records(iterable: Iterable[Any], start: int, end: int, step: int, max_records: int) -> Iterable[Any]:
    start = 0 if start is None else start
    stop = end
    step = 1 if step is None else step
    sliced = itertools.islice(iterable, start, stop, step)
    if max_records is not None:
        sliced = itertools.islice(sliced, max_records)
    return sliced
