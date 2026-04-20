from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Iterable, Optional, Set

from dp.loaders import get_adapter


def _normalize_names(values: Iterable[str]) -> Set[str]:
    names: Set[str] = set()
    for value in values:
        text = str(value).strip()
        if text:
            names.add(text)
    return names


def _collect_names(dataset: str, data_in: str, max_records: Optional[int] = None) -> Set[str]:
    adapter = get_adapter(dataset, data=dataset, data_in=data_in, max_records=max_records)
    names: Set[str] = set()
    for record in adapter.iter_records():
        name = str(record.name).strip()
        if name:
            names.add(name)
    return names


class SensitiveIndividualSelector(ABC):
    @abstractmethod
    def select_individuals(self, total_individuals: Iterable[str], *args: Any, **kwargs: Any) -> Set[str]:
        raise NotImplementedError


class TabSensitiveIndividualSelector(SensitiveIndividualSelector):
    def select_individuals(self, total_individuals: Iterable[str], subset_file: str, subset_max_records: Optional[int] = None) -> Set[str]:
        subset_file = str(Path(str(subset_file)).expanduser())
        source_names = _collect_names("tab", subset_file, subset_max_records)
        total = _normalize_names(total_individuals)
        return total.intersection(source_names)


class DBBioSensitiveIndividualSelector(SensitiveIndividualSelector):
    def select_individuals(self, total_individuals: Iterable[str], subset_file: str, subset_max_records: Optional[int] = None) -> Set[str]:
        subset_file = str(Path(str(subset_file)).expanduser())
        source_names = _collect_names("db_bio", subset_file, subset_max_records)
        total = _normalize_names(total_individuals)
        return total.intersection(source_names)


SENSITIVE_SELECTOR_REGISTRY: dict[str, type[SensitiveIndividualSelector]] = {
    "tab": TabSensitiveIndividualSelector,
    "db_bio": DBBioSensitiveIndividualSelector,
}


def get_sensitive_selector(name: str) -> SensitiveIndividualSelector:
    key = str(name).strip().lower()
    if key not in SENSITIVE_SELECTOR_REGISTRY:
        raise ValueError(
            f"Unknown sensitive selector '{name}'. "
            f"Available selectors: {sorted(SENSITIVE_SELECTOR_REGISTRY.keys())}"
        )
    return SENSITIVE_SELECTOR_REGISTRY[key]()

