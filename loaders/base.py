"""Base classes and attacker adapter for dataset processing."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple
from pathlib import Path
import json
from tqdm import tqdm

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
    spans: Optional[list[TextAnnotation]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class DatasetAdapter:
    """Base adapter providing a unified interface across datasets."""

    data: Optional[str] = None
    data_in: Optional[str] = None
    max_records: Optional[int] = None

    def __iter__(self) -> Iterator[DatasetRecord]:
        return iter(self.iter_records())

    def iter_records(self, *args, **kwargs) -> Iterable[DatasetRecord]:
        """Yield normalized dataset records."""
        raise NotImplementedError

    def __len__(self) -> int:
        """Return the size of the dataset if known."""
        raise NotImplementedError

@dataclass
class AttackerDatasetRecord(DatasetRecord):
    background_knowledge: List[Tuple[str, str]] = field(default_factory=list)
    summarized_text: Optional[str] = None

# Attacker adapter lives under dp.tri.attacker_adapter to keep loaders lean and avoid circular deps.