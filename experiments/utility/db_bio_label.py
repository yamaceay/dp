from __future__ import annotations

from typing import List, Optional

from dp.loaders.base import DatasetRecord
from experiments.utility.base import TextUtilityExperiment


class DBBioLabelExperiment(TextUtilityExperiment):
    def __init__(
        self,
        records: List[DatasetRecord],
        max_records: Optional[int] = None,
        test_size: float = 0.2,
        random_state: int = 42,
    ):
        super().__init__(records=records, max_records=max_records, test_size=test_size, random_state=random_state)

    def get_label(self, record: DatasetRecord) -> Optional[str]:
        value = record.utilities.get("label")
        if value is None:
            return None
        return str(value)
