from __future__ import annotations

from pathlib import Path
from typing import Callable, Iterable, Optional

from dp.loaders import get_adapter
from dp.loaders.base import DatasetRecord
from experiments.utility.base import TextUtilityExperiment

REGION_GROUPS = {
    "northern_europe": {"DNK", "NOR", "SWE", "FIN", "ISL"},
    "western_europe": {"FRA", "DEU", "BEL", "NLD", "GBR", "IRL", "LUX", "CHE", "AUT"},
    "southern_europe": {"ESP", "PRT", "ITA", "GRC", "MLT", "CYP", "SMR", "AND"},
    "central_europe": {"POL", "CZE", "SVK", "HUN", "SVN"},
    "baltic": {"EST", "LVA", "LTU"},
    "eastern_europe": {"RUS", "UKR", "BLR", "MDA"},
    "balkans": {"ROU", "BGR", "SRB", "HRV", "BIH", "MKD", "MNE", "ALB", "KSV"},
    "caucasus": {"ARM", "AZE", "GEO"},
    "middle_east": {"TUR"},
}


def group_country(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    key = value.upper()
    for region, codes in REGION_GROUPS.items():
        if key in codes:
            return region
    return "other"


def group_year(value: Optional[int]) -> Optional[str]:
    if value is None:
        return None
    try:
        year = int(value)
    except (TypeError, ValueError):
        return None
    decade = (year // 10) * 10
    return f"{decade}s"


class TabMetadataExperiment(TextUtilityExperiment):
    def __init__(
        self,
        data_file: str,
        target: str,
        anonymize: Callable[[str], str],
        max_records: Optional[int] = None,
        test_size: float = 0.2,
        random_state: int = 42,
    ):
        if target not in {"country_region", "year_decade"}:
            raise ValueError("Unsupported target")
        super().__init__(anonymize, max_records=max_records, test_size=test_size, random_state=random_state)
        self.data_file = Path(data_file)
        self.target = target

    def load_records(self) -> Iterable[DatasetRecord]:
        adapter = get_adapter("tab", data_in=str(self.data_file))
        yield from adapter.iter_records()

    def get_label(self, record: DatasetRecord) -> Optional[str]:
        meta = record.metadata.get("meta", {})
        if self.target == "country_region":
            return group_country(meta.get("countries"))
        if self.target == "year_decade":
            return group_year(meta.get("year"))
        return None
