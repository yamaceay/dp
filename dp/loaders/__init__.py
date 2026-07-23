"""
Dataset adapter interfaces and helpers for DPMLM benchmarking.

Adapters provide a consistent way to access dataset records with a unique
identifier, raw text, optional annotations, and optional utility metadata.
"""

from dp.loaders.base import DatasetAdapter, DatasetRecord, TextAnnotation, TextAnnotations, TokenEdit

from dp.loaders._tab import TabDatasetAdapter
from dp.loaders._reddit import RedditDatasetAdapter
from dp.loaders._yelp import YelpDatasetAdapter

try:
    from dp.loaders._dbbio import DBBioDatasetAdapter
except ModuleNotFoundError:  # Optional dependency: `datasets`
    DBBioDatasetAdapter = None  # type: ignore[assignment]

try:
    from dp.loaders._ratbench import RatBenchDatasetAdapter
except ModuleNotFoundError:  # Optional dependency: `datasets`
    RatBenchDatasetAdapter = None  # type: ignore[assignment]

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
from dp.loaders.results import build_dataset_from_results


ADAPTER_REGISTRY: dict[str, type[DatasetAdapter]] = {
    "tab": TabDatasetAdapter,
    "reddit": RedditDatasetAdapter,
    "yelp": YelpDatasetAdapter,
}
if DBBioDatasetAdapter is not None:
    ADAPTER_REGISTRY["db_bio"] = DBBioDatasetAdapter
if RatBenchDatasetAdapter is not None:
    ADAPTER_REGISTRY["rat_bench"] = RatBenchDatasetAdapter

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
    "build_dataset_from_results",
    "ADAPTER_REGISTRY",
]
