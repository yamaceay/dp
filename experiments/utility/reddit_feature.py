from __future__ import annotations

from typing import List, Optional

from dp.loaders.base import DatasetRecord
from experiments.utility.base import TextUtilityExperiment


class RedditUtilityExperiment(TextUtilityExperiment):
    def __init__(
        self,
        records: List[DatasetRecord],
        target: str = "label",
        max_records: Optional[int] = None,
        test_size: float = 0.2,
        random_state: int = 42,
    ):
        if target not in {"label", "feature"}:
            raise ValueError("target must be 'label' or 'feature'")
        super().__init__(records=records, max_records=max_records, test_size=test_size, random_state=random_state)
        self.target = target

    def get_label(self, record: DatasetRecord) -> Optional[str]:
        value = record.utilities.get(self.target)
        if value is None:
            return None
        return str(value)
