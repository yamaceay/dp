from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

from dp.loaders.base import DatasetRecord


def text_value(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, str):
        t = value.strip()
        return t or None
    if isinstance(value, (list, tuple, set)):
        for item in value:
            t = text_value(item)
            if t:
                return t
        return None
    t = str(value).strip()
    return t or None


def int_value(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def float_value(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def group_income(label: str) -> str:
    if label in {"very high", "high"}:
        return "high"
    if label == "middle":
        return "middle"
    return "low"


def _country_from_city_country(label: str) -> str:
    if "," in label:
        tail = label.split(",", 1)[1].strip()
        us_states = {"Alabama","Alaska","Arizona","Arkansas","California","Colorado","Connecticut","Delaware","Florida","Georgia","Hawaii","Idaho","Illinois","Indiana","Iowa","Kansas","Kentucky","Louisiana","Maine","Maryland","Massachusetts","Michigan","Minnesota","Mississippi","Missouri","Montana","Nebraska","Nevada","New Hampshire","New Jersey","New Mexico","New York","North Carolina","North Dakota","Ohio","Oklahoma","Oregon","Pennsylvania","Rhode Island","South Carolina","South Dakota","Tennessee","Texas","Utah","Vermont","Virginia","Washington","West Virginia","Wisconsin","Wyoming"}
        if tail in us_states:
            return "United States"
        return tail
    return label.strip()


def group_region(label: str) -> str:
    country = _country_from_city_country(label)
    europe = {"United Kingdom", "Ireland", "Sweden", "Finland", "France", "Netherlands", "Italy", "Spain", "Hungary", "Portugal", "Germany", "Norway"}
    americas = {"United States", "Canada", "Mexico", "Brazil", "Argentina", "Colombia"}
    asia = {"India", "Japan", "China", "Turkey"}
    africa = {"South Africa", "Zambia"}
    oceania = {"Australia", "New Zealand"}
    if country in europe:
        return "Europe"
    if country in americas:
        return "Americas"
    if country in asia:
        return "Asia"
    if country in africa:
        return "Africa"
    if country in oceania:
        return "Oceania"
    return "Other"


def group_age(label: str) -> str:
    age = int(label)
    if 18 <= age <= 29:
        return "18-29"
    if 30 <= age <= 44:
        return "30-44"
    if 45 <= age <= 59:
        return "45-59"
    return "60+"


def group_education(label: str) -> str:
    l = label.lower()
    if "high school" in l:
        return "secondary"
    if "studying" in l or "currently studying" in l:
        return "studying"
    if l.startswith("bachelors") or "diploma" in l:
        return "bachelor"
    if l.startswith("masters") or "law degree" in l:
        return "master"
    if l.startswith("phd") or "doctorate" in l:
        return "doctorate"
    return "other"


def group_occupation(label: str) -> str:
    l = label.lower()
    tech = {"software engineer", "junior software developer", "web developer", "game developer", "data scientist"}
    design = {"graphic designer", "part-time graphic designer", "part-time film editor"}
    education = {"university professor", "college professor", "high school principal", "part-time tutor"}
    healthcare = {"surgeon", "nurse"}
    business_finance = {"financial manager", "financial analyst", "retired ceo", "business development manager", "lawyer"}
    service = {"chef", "part-time retail worker"}
    culture = {"museum curator", "art curator"}
    if l in tech:
        return "tech"
    if l in design:
        return "design"
    if l in education:
        return "education"
    if l in healthcare:
        return "healthcare"
    if l in business_finance:
        return "business_finance"
    if l in service:
        return "service"
    if l in culture:
        return "culture"
    return "other"


def group_relationship(label: str) -> str:
    return label


def group_sex(label: str) -> str:
    return label


GROUPERS: Dict[str, Callable[[str], str]] = {
    "income_level": group_income,
    "birth_city_country": group_region,
    "city_country": group_region,
    "age": group_age,
    "education": group_education,
    "occupation": group_occupation,
    "relationship_status": group_relationship,
    "sex": group_sex,
}


def reddit_feature(record: DatasetRecord) -> Optional[str]:
    return text_value(record.metadata.get("feature"))


def reddit_label(record: DatasetRecord) -> Optional[str]:
    feature = reddit_feature(record)
    return text_value(record.metadata.get(f"persona_{feature}"))


def reddit_feature_label(record: DatasetRecord, group: bool = True) -> Optional[str]:
    feature = reddit_feature(record)
    label = reddit_label(record)
    if feature is None or label is None:
        return None
    if group and feature in GROUPERS:
        label = GROUPERS[feature](label)
    return f"{feature}: {label}"


def tab_country(record: DatasetRecord) -> Optional[str]:
    return text_value(record.metadata.get("country"))


def tab_year(record: DatasetRecord) -> Optional[int]:
    value = record.metadata.get("years")
    if isinstance(value, (list, tuple, set)):
        for item in value:
            number = int_value(item)
            if number is not None:
                return number
        return None
    return int_value(value)


def tab_year_groups(record: DatasetRecord, year_groups: List[str] | None = None) -> str:
    groups = year_groups or [
        "1984-1995","1996-1998","1999-2001","2002-2004","2005-2007","2008-2010","2011-2013","2014-2019"
    ]
    year = record.metadata.get("years")
    bounds = [(int(g.split("-")[0]), int(g.split("-")[1])) for g in groups]
    mapping = {c: g for g, (start, end) in zip(groups, bounds) for c in range(start, end + 1)}
    if year not in mapping:
        raise ValueError(f"Year {year} not in any defined year groups.")
    return mapping[year]


def tab_country_groups(record: DatasetRecord, region_groups: List[str] | None = None) -> str:
    groups = region_groups or ["GBR-IRL", "SWE-NOR-DNK"]
    region = record.metadata.get("country")
    mapping = {c: g for g in groups for c in g.split("-")}
    return mapping.get(region, region)


def db_bio_label(record: DatasetRecord) -> Optional[str]:
    return text_value(record.metadata.get("label"))


def trustpilot_category(record: DatasetRecord) -> Optional[str]:
    return text_value(record.metadata.get("category"))


def trustpilot_stars(record: DatasetRecord) -> Optional[int]:
    return int_value(record.metadata.get("stars"))


DERIVE_REGISTRY: Dict[str, Dict[str, Callable[[DatasetRecord], Any]]] = {
    "reddit": {
        "feature": reddit_feature,
        "label": reddit_label,
        "feature_label": reddit_feature_label,
        "feature_label_exact": lambda r: reddit_feature_label(r, group=False),
    },
    "tab": {
        "country": tab_country,
        "year": tab_year,
        "year_group": tab_year_groups,
        "country_group": tab_country_groups,
    },
    "db_bio": {
        "label": db_bio_label,
    },
    "trustpilot": {
        "category": trustpilot_category,
        "stars": trustpilot_stars,
    },
}


def get_getter(dataset: str, key: str) -> Callable[[DatasetRecord], Any]:
    if dataset not in DERIVE_REGISTRY:
        raise ValueError(f"Unknown dataset '{dataset}' for derive getters.")
    if key not in DERIVE_REGISTRY[dataset]:
        raise ValueError(f"Unknown key '{key}' for dataset '{dataset}' in derive getters.")
    return DERIVE_REGISTRY[dataset][key]