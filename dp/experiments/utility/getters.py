from __future__ import annotations

from typing import Any, Dict, Optional

from dp.loaders.base import DatasetRecord
from dp.experiments.utility.base import UtilityTarget

def _text_value(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip()
        return text or None
    if isinstance(value, (list, tuple, set)):
        for item in value:
            text = _text_value(item)
            if text:
                return text
        return None
    text = str(value).strip()
    return text or None


def _int_value(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _float_value(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _reddit_feature(record: DatasetRecord) -> Optional[str]:
    return _text_value(record.metadata.get("feature"))


def _reddit_label(record: DatasetRecord) -> Optional[str]:
    return _text_value(record.metadata.get("label"))


def _reddit_age(record: DatasetRecord) -> Optional[int]:
    return _int_value(record.metadata.get("persona_age"))


def _reddit_age_group(record: DatasetRecord) -> Optional[str]:
    age = _reddit_age(record)
    if age is None:
        return None
    if age < 18:
        return "under_18"
    if age < 25:
        return "18_24"
    if age < 35:
        return "25_34"
    if age < 45:
        return "35_44"
    if age < 55:
        return "45_54"
    if age < 65:
        return "55_64"
    return "65_plus"


def _reddit_sex(record: DatasetRecord) -> Optional[str]:
    return _text_value(record.metadata.get("persona_sex"))


def _reddit_income(record: DatasetRecord) -> Optional[str]:
    return _text_value(record.metadata.get("persona_income_level"))


def _tab_country(record: DatasetRecord) -> Optional[str]:
    return _text_value(record.metadata.get("country"))


def _tab_year(record: DatasetRecord) -> Optional[int]:
    value = record.metadata.get("years")
    if isinstance(value, (list, tuple, set)):
        for item in value:
            number = _int_value(item)
            if number is not None:
                return number
        return None
    return _int_value(value)


def _tab_year_groups(record, year_groups=['1990-1995', '1996-1998', '1999-2001', '2002-2004', '2005-2007', '2008-2010', '2011-2013', '2014-2019']):
    year = record.metadata.get('years', None)
    year_bounds = [(int(group.split('-')[0]), int(group.split('-')[1])) for group in year_groups]
    yeargroup_map = {c: group for group, (start, end) in zip(year_groups, year_bounds) for c in range(start, end + 1)}
    if year not in yeargroup_map:
        raise ValueError(f"Year {year} not in any defined year groups.")
    return yeargroup_map[year]

def _tab_country_groups(record, region_groups=['GBR-IRL', 'SWE-NOR-DNK']):
    region = record.metadata.get('country', None)
    region_map = {c: group for group in region_groups for c in group.split('-')}
    return region_map.get(region, region)


def _db_bio_label(record: DatasetRecord) -> Optional[str]:
    return _text_value(record.metadata.get("label"))


def _trustpilot_category(record: DatasetRecord) -> Optional[str]:
    return _text_value(record.metadata.get("category"))


def _trustpilot_stars(record: DatasetRecord) -> Optional[int]:
    return _int_value(record.metadata.get("stars"))

UTILITY_TARGETS: Dict[str, Dict[str, UtilityTarget]] = {
    "reddit": {
        "feature": UtilityTarget(name="feature", source="reddit", mode=UtilityTarget.Mode.NOMINAL, getter=_reddit_feature),
        "label": UtilityTarget(name="label", source="reddit", mode=UtilityTarget.Mode.NOMINAL, getter=_reddit_label),
        "age_group": UtilityTarget(name="age_group", source="reddit", mode=UtilityTarget.Mode.NOMINAL, getter=_reddit_age_group),
    },
    "tab": {
        "country": UtilityTarget(name="country", source="tab", mode=UtilityTarget.Mode.NOMINAL, getter=_tab_country),
        "year": UtilityTarget(name="year", source="tab", mode=UtilityTarget.Mode.CARDINAL, getter=_tab_year),
        "year_group": UtilityTarget(name="year_group", source="tab", mode=UtilityTarget.Mode.NOMINAL, getter=_tab_year_groups),
        "country_group": UtilityTarget(name="country_group", source="tab", mode=UtilityTarget.Mode.NOMINAL, getter=_tab_country_groups),
    },
    "db_bio": {
        "label": UtilityTarget(name="label", source="db_bio", mode=UtilityTarget.Mode.NOMINAL, getter=_db_bio_label),
    },
    "trustpilot": {
        "category": UtilityTarget(name="category", source="trustpilot", mode=UtilityTarget.Mode.NOMINAL, getter=_trustpilot_category),
        "stars": UtilityTarget(name="stars", source="trustpilot", mode=UtilityTarget.Mode.CARDINAL, getter=_trustpilot_stars),
    },
}
