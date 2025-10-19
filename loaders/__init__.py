"""
Dataset adapter interfaces and helpers for DPMLM benchmarking.

Adapters provide a consistent way to access dataset records with a unique
identifier, raw text, optional annotations, and optional utility metadata.
"""

from dp.loaders.base import DatasetAdapter, DatasetRecord, TextAnnotation
from dp.loaders.trustpilot import TrustpilotDatasetAdapter
from dp.loaders.tab import TabDatasetAdapter
from dp.loaders.db_bio import DBBioDatasetAdapter
from dp.loaders.annotations import (
    read_annotations,
    write_annotations,
    apply_annotations,
    annotations_to_spans,
    spans_to_annotations
)


ADAPTER_REGISTRY: dict[str, type[DatasetAdapter]] = {
    "trustpilot": TrustpilotDatasetAdapter,
    "tab": TabDatasetAdapter,
    "db_bio": DBBioDatasetAdapter,
}


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
    "TrustpilotDatasetAdapter",
    "TabDatasetAdapter",
    "DBBioDatasetAdapter",
    "TextAnnotation",
    "get_adapter",
    "read_annotations",
    "write_annotations",
    "apply_annotations",
    "annotations_to_spans",
    "spans_to_annotations",
]
