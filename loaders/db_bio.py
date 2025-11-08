"""DB-Bio dataset adapter with flexible split discovery."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional, Union

from datasets import Dataset, DatasetDict, load_dataset

from dp.loaders.base import DatasetAdapter, DatasetRecord

class DBBioDatasetAdapter(DatasetAdapter):
    """Adapter for the DB-Bio legal dataset."""

    def __init__(
        self,
        data: Optional[str] = None,
        data_in: Optional[str] = None,
        max_records: Optional[int] = None,
    ):
        self.data_in = Path(data_in)
        self.max_records = max_records

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

    def __len__(self) -> int:
        if isinstance(self._dataset, DatasetDict):
            return sum(len(split) for split in self._dataset.values())
        return len(self._dataset)

    def iter_records(self) -> Iterable[DatasetRecord]:
        yielded = 0
        if isinstance(self._dataset, DatasetDict):
            for split_name in self._ordered_split_names(self._dataset):
                split_dataset = self._dataset[split_name]
                for record in self._iter_split(split_dataset, split_name):
                    yield record
                    yielded += 1
                    if self.max_records is not None and yielded >= self.max_records:
                        return
        else:
            for record in self._iter_split(self._dataset, None):
                yield record
                yielded += 1
                if self.max_records is not None and yielded >= self.max_records:
                    return

    def _iter_split(self, dataset: Dataset, split_name: Optional[str]) -> Iterable[DatasetRecord]:
        for idx, row in enumerate(dataset):
            if isinstance(row, dict):
                data = row
            else:
                data = dict(row)
            text = data.get("text", "")
            uid = data.get("wiki_name") or data.get("label") or str(idx)
            name = data.get("people")
            metadata = {
                "label": data.get("label"),
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
                name=name,
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

__all__ = ["DBBioDatasetAdapter"]
