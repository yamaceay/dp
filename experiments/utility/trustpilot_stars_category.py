from __future__ import annotations

from pathlib import Path
from typing import Callable, Iterable, Optional

from dp.loaders import get_adapter
from dp.loaders.base import DatasetRecord
from experiments.utility.base import TextUtilityExperiment
from dp.loaders import TrustpilotDatasetAdapter

class TrustpilotStarsExperiment(TextUtilityExperiment):
    def __init__(
        self,
        data_dir: str,
        anonymize: Callable[[str], str],
        max_records: Optional[int] = None,
        test_size: float = 0.2,
        random_state: int = 42,
    ):
        super().__init__(anonymize, max_records=max_records, test_size=test_size, random_state=random_state)
        self.data_dir = Path(data_dir)

    def load_records(self) -> Iterable[DatasetRecord]:
        for path in sorted(self.data_dir.iterdir()):
            if not path.is_dir():
                continue
            data_in = path / "train.json"
            if not data_in.exists():
                continue
            adapter = TrustpilotDatasetAdapter(data="trustpilot", data_in=str(data_in))
            yield from adapter.iter_records()

    def get_label(self, record: DatasetRecord) -> Optional[str]:
        value = record.utilities.get("stars")
        if value is None:
            return None
        return str(value)
