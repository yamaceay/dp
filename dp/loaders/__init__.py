"""
Dataset adapter interfaces and helpers for DPMLM benchmarking.

Adapters provide a consistent way to access dataset records with a unique
identifier, raw text, optional annotations, and optional utility metadata.
"""

from dp.loaders.base import DatasetAdapter, DatasetRecord, TextAnnotation, TextAnnotations, TokenEdit
from dp.loaders.tab import TabDatasetAdapter
from dp.loaders.db_bio import DBBioDatasetAdapter
from dp.loaders.reddit import RedditDatasetAdapter

try:
    from dp.loaders.trustpilot_company import TrustpilotDatasetAdapter
except ImportError:
    TrustpilotDatasetAdapter = None
from dp.loaders.annotations import (
    read_annotations,
    write_annotations,
    apply_annotations,
    annotations_to_spans,
    spans_to_annotations,
    read_batch_annotations,
    read_batch_annotations_from_path,
    read_batch_textannotations,
    read_batch_textannotations_from_path,
    list_batch_timestamps,
)


ADAPTER_REGISTRY: dict[str, type[DatasetAdapter]] = {
    "tab": TabDatasetAdapter,
    "db_bio": DBBioDatasetAdapter,
    "reddit": RedditDatasetAdapter,
}

if TrustpilotDatasetAdapter is not None:
    ADAPTER_REGISTRY["trustpilot"] = TrustpilotDatasetAdapter


def get_adapter(name: str, **kwargs) -> DatasetAdapter:
    """Instantiate a dataset adapter by name."""
    key = (name or "").lower()
    if key not in ADAPTER_REGISTRY:
        raise ValueError(
            f"Unknown dataset adapter '{name}'. "
            f"Available adapters: {sorted(ADAPTER_REGISTRY.keys())}"
        )
    adapter_cls = ADAPTER_REGISTRY[key]
    return adapter_cls(**kwargs)

__all__ = [
    "DatasetAdapter",
    "DatasetRecord",
    "TabDatasetAdapter",
    "DBBioDatasetAdapter",
    "TextAnnotation",
    "TextAnnotations",
    "TokenEdit",
    "get_adapter",
    "read_annotations",
    "write_annotations",
    "apply_annotations",
    "annotations_to_spans",
    "spans_to_annotations",
    "read_batch_annotations",
    "read_batch_annotations_from_path",
    "read_batch_textannotations",
    "read_batch_textannotations_from_path",
    "list_batch_timestamps",
    "ADAPTER_REGISTRY",
]

if TrustpilotDatasetAdapter is not None:
    __all__.append("TrustpilotDatasetAdapter")
