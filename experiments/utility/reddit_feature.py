from __future__ import annotations

from typing import Callable, Iterable, Optional

from dp.loaders import get_adapter
from dp.loaders.base import DatasetRecord
from experiments.utility.base import TextUtilityExperiment


class RedditUtilityExperiment(TextUtilityExperiment):
    def __init__(
        self,
        data_path: str,
        anonymize: Callable[[str], str],
        target: str = "label",
        max_records: Optional[int] = None,
        test_size: float = 0.2,
        random_state: int = 42,
    ):
        if target not in {"label", "feature"}:
            raise ValueError("target must be 'label' or 'feature'")
        super().__init__(anonymize, max_records=max_records, test_size=test_size, random_state=random_state)
        self.data_path = data_path
        self.target = target

    def load_records(self) -> Iterable[DatasetRecord]:
        adapter = get_adapter("reddit", data_in=self.data_path, max_records=self.max_records)
        yield from adapter.iter_records()

    def get_label(self, record: DatasetRecord) -> Optional[str]:
        value = record.utilities.get(self.target)
        if value is None:
            return None
        return str(value)
