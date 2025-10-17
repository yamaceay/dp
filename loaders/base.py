"""Base classes for dataset adapters used in benchmarking."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Iterator, Optional

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

    uid: str
    text: str
    name: str = ""
    spans: Optional[list[TextAnnotation]] = None
    utilities: Dict[str, Optional[Any]] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


class DatasetAdapter:
    """Base adapter providing a unified interface across datasets."""

    data: Optional[str] = None
    data_in: Optional[str] = None
    max_records: Optional[int] = None


    def __iter__(self) -> Iterator[DatasetRecord]:
        return iter(self.iter_records())

    def iter_records(self) -> Iterable[DatasetRecord]:
        """Yield normalized dataset records."""
        raise NotImplementedError

    def __len__(self) -> int:
        """Return the size of the dataset if known."""
        raise NotImplementedError
