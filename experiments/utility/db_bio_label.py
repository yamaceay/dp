from __future__ import annotations

from pathlib import Path
from typing import Callable, Iterable, Optional

from dp.loaders import get_adapter
from dp.loaders.base import DatasetRecord
from experiments.utility.base import TextUtilityExperiment


class DBBioLabelExperiment(TextUtilityExperiment):
    def __init__(
        self,
        data_dir: str,
        split: str,
        anonymize: Callable[[str], str],
        max_records: Optional[int] = None,
        test_size: float = 0.2,
        random_state: int = 42,
    ):
        super().__init__(anonymize, max_records=max_records, test_size=test_size, random_state=random_state)
        self.data_dir = Path(data_dir)
        self.split = split

    def load_records(self) -> Iterable[DatasetRecord]:
        split_dir = self.data_dir / self.split
        if not split_dir.exists():
            raise ValueError(f"Missing split directory: {split_dir}")
        arrow_files = sorted(split_dir.glob("*.arrow"))
        if not arrow_files:
            raise ValueError(f"No Arrow files found in {split_dir}")
        adapter = get_adapter("db_bio", data_in=str(arrow_files[0]))
        yield from adapter.iter_records()

    def get_label(self, record: DatasetRecord) -> Optional[str]:
        value = record.utilities.get("label")
        if value is None:
            return None
        return str(value)
