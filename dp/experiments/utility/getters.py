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


def _reddit_age(record: DatasetRecord) -> Optional[float]:
    return _float_value(record.metadata.get("persona_age"))


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


def _tab_year(record: DatasetRecord) -> Optional[float]:
    value = record.metadata.get("years")
    if isinstance(value, (list, tuple, set)):
        for item in value:
            number = _float_value(item)
            if number is not None:
                return number
        return None
    return _float_value(value)


def _db_bio_label(record: DatasetRecord) -> Optional[str]:
    return _text_value(record.metadata.get("label"))


def _trustpilot_category(record: DatasetRecord) -> Optional[str]:
    return _text_value(record.metadata.get("category"))


def _trustpilot_stars(record: DatasetRecord) -> Optional[float]:
    return _float_value(record.metadata.get("stars"))

UTILITY_TARGETS: Dict[str, Dict[str, UtilityTarget]] = {
    "reddit": {
        "feature": UtilityTarget(name="feature", source="reddit", mode=UtilityTarget.Mode.NOMINAL, getter=_reddit_feature),
        "label": UtilityTarget(name="label", source="reddit", mode=UtilityTarget.Mode.NOMINAL, getter=_reddit_label),
        "age": UtilityTarget(name="age", source="reddit", mode=UtilityTarget.Mode.CARDINAL, getter=_reddit_age),
        "age_group": UtilityTarget(name="age_group", source="reddit", mode=UtilityTarget.Mode.NOMINAL, getter=_reddit_age_group),
        "sex": UtilityTarget(name="sex", source="reddit", mode=UtilityTarget.Mode.BINARY, getter=_reddit_sex),
        "income_level": UtilityTarget(name="income_level", source="reddit", mode=UtilityTarget.Mode.ORDINAL, getter=_reddit_income),
    },
    "tab": {
        "country": UtilityTarget(name="country", source="tab", mode=UtilityTarget.Mode.NOMINAL, getter=_tab_country),
        "year": UtilityTarget(name="year", source="tab", mode=UtilityTarget.Mode.CARDINAL, getter=_tab_year),
    },
    "db_bio": {
        "label": UtilityTarget(name="label", source="db_bio", mode=UtilityTarget.Mode.NOMINAL, getter=_db_bio_label),
    },
    "trustpilot": {
        "category": UtilityTarget(name="category", source="trustpilot", mode=UtilityTarget.Mode.NOMINAL, getter=_trustpilot_category),
        "stars": UtilityTarget(name="stars", source="trustpilot", mode=UtilityTarget.Mode.CARDINAL, getter=_trustpilot_stars),
    },
}
