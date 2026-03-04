"""DB-Bio dataset adapter with flexible split discovery."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional, Union

from datasets import Dataset, DatasetDict, load_dataset

from dp.loaders.base import DatasetAdapter, DatasetRecord, load_split_indices

class DBBioDatasetAdapter(DatasetAdapter):
    """Adapter for the DB-Bio legal dataset."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.data_in = Path(self.data_in)
        try:
            if self.data_in.is_dir():
                data_files = self._discover_split_files(self.data_in)
                dataset = load_dataset("arrow", data_files=data_files)
                self._dataset: Union[Dataset, DatasetDict] = dataset
            else:
                dataset = load_dataset("arrow", data_files={"data": str(self.data_in)})
                self._dataset = dataset["data"]
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(f"Failed to load DB-Bio dataset from {self.data_in}") from exc
        self._split_indices = self._resolve_split_indices()

    def __len__(self) -> int:
        if self._split_indices is not None:
            return len(self._split_indices)
        if isinstance(self._dataset, DatasetDict):
            return sum(len(split) for split in self._dataset.values())
        return len(self._dataset)

    def iter_records(self) -> Iterable[DatasetRecord]:
        all_records = self._iter_all_records()
        if self._split_indices is None:
            for record in self._slice_records(all_records):
                yield record
            return
        records_list = list(all_records)
        total = len(records_list)
        for idx in self._split_indices:
            if idx >= total:
                raise ValueError(f"Split index {idx} out of range for DB-Bio dataset (size={total})")
        split_records = ((idx, records_list[idx]) for idx in self._split_indices)
        for _, record in self._slice_records(split_records):
            yield record

    def _iter_all_records(self) -> Iterable[DatasetRecord]:
        if isinstance(self._dataset, DatasetDict):
            for split_name in self._ordered_split_names(self._dataset):
                split_dataset = self._dataset[split_name]
                for record in self._iter_split(split_dataset, split_name):
                    yield record
        else:
            for record in self._iter_split(self._dataset, None):
                yield record

    def _iter_split(self, dataset: Dataset, split_name: Optional[str]) -> Iterable[DatasetRecord]:
        for idx, row in enumerate(dataset):
            if isinstance(row, dict):
                data = row
            else:
                data = dict(row)
            text = data.get("text", "")
            uid = data.get("wiki_name")
            name = data.get("people")
            label = data.get("label")
            if uid is None or name is None or label is None:
                raise ValueError(f"Missing required fields in DB-Bio record at index {idx}")
            name_unique = f"{name} ({label})"
            metadata = {
                "label": label,
                "l1": data.get("l1"),
                "l2": data.get("l2"),
                "l3": data.get("l3"),
                "word_count": data.get("word_count"),
                "wiki_name": data.get("wiki_name"),
            }
            if split_name:
                metadata["split"] = split_name
            yield DatasetRecord(
                text=text,
                uid=str(uid),
                name=name_unique,
                metadata=metadata,
            )

    def _discover_split_files(self, folder: Path) -> Dict[str, Union[str, List[str]]]:
        data_files: Dict[str, Union[str, List[str]]] = {}
        for split_name in ("train", "validation", "test"):
            split_dir = folder / split_name
            if not split_dir.exists():
                continue
            arrow_files = sorted(split_dir.glob("*.arrow"))
            if not arrow_files:
                continue
            if len(arrow_files) == 1:
                data_files[split_name] = str(arrow_files[0])
            else:
                data_files[split_name] = [str(f) for f in arrow_files]
        if data_files:
            return data_files
        arrow_files = sorted(folder.glob("*.arrow"))
        if not arrow_files:
            raise RuntimeError(f"No Arrow files found in {folder}")
        return {"data": [str(f) for f in arrow_files]}

    def _ordered_split_names(self, dataset: DatasetDict) -> List[str]:
        preferred = ["train", "validation", "test"]
        ordered = [name for name in preferred if name in dataset]
        ordered.extend(name for name in dataset if name not in preferred)
        return ordered

    def _resolve_split_indices(self) -> Optional[List[int]]:
        if self.split is not None:
            return load_split_indices(data_name=self.data, split=self.split)
        return self._load_default_indices()

    def _load_default_indices(self) -> List[int]:
        split_names = ["train", "val", "test"]
        concatenated: List[int] = []
        for split_name in split_names:
            indices = load_split_indices(data_name="db_bio", split=split_name)
            if indices is None:
                raise ValueError(f"Missing default split indices for db_bio: {split_name}")
            concatenated.extend(indices)
        return self._stable_unique(concatenated)

    def _stable_unique(self, values: List[int]) -> List[int]:
        seen: set[int] = set()
        unique_values: List[int] = []
        for value in values:
            if value in seen:
                continue
            seen.add(value)
            unique_values.append(value)
        return unique_values

__all__ = ["DBBioDatasetAdapter"]
