from __future__ import annotations

from typing import Dict
from dp.experiments.utility.base import UtilityTarget
from dp.loaders.derive import get_getter

UTILITY_TARGETS: Dict[str, Dict[str, UtilityTarget]] = {
    "reddit": {
        "label": UtilityTarget(name="label", source="reddit", mode=UtilityTarget.Mode.NOMINAL, getter=get_getter("reddit", "label")),
        "feature_label": UtilityTarget(name="feature_label", source="reddit", mode=UtilityTarget.Mode.NOMINAL, getter=get_getter("reddit", "feature_label")),
        "feature_label_exact": UtilityTarget(name="feature_label_exact", source="reddit", mode=UtilityTarget.Mode.NOMINAL, getter=get_getter("reddit", "feature_label_exact")),
    },
    "tab": {
        "country": UtilityTarget(name="country", source="tab", mode=UtilityTarget.Mode.NOMINAL, getter=get_getter("tab", "country")),
        "year": UtilityTarget(name="year", source="tab", mode=UtilityTarget.Mode.CARDINAL, getter=get_getter("tab", "year")),
        "year_group": UtilityTarget(name="year_group", source="tab", mode=UtilityTarget.Mode.NOMINAL, getter=get_getter("tab", "year_group")),
        "country_group": UtilityTarget(name="country_group", source="tab", mode=UtilityTarget.Mode.NOMINAL, getter=get_getter("tab", "country_group")),
    },
    "db_bio": {
        "label": UtilityTarget(name="label", source="db_bio", mode=UtilityTarget.Mode.NOMINAL, getter=get_getter("db_bio", "label")),
    },
    "trustpilot": {
        "category": UtilityTarget(name="category", source="trustpilot", mode=UtilityTarget.Mode.NOMINAL, getter=get_getter("trustpilot", "category")),
        "stars": UtilityTarget(name="stars", source="trustpilot", mode=UtilityTarget.Mode.CARDINAL, getter=get_getter("trustpilot", "stars")),
    },
}
