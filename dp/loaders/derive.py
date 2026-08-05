from __future__ import annotations

import re
from typing import Any, Callable, Dict, List, Optional, Tuple

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
    raise ValueError(f"Unknown education label: {label}")


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

ORDINAL_GROUPERS: Dict[str, Tuple[Callable[[str], str], List[str]]] = {
    "income_level": (group_income, ["low", "middle", "high"]),
    "age": (group_age, ["18-29", "30-44", "45-59", "60+"]),
    "education": (group_education, ["secondary", "studying", "bachelor", "master", "doctorate", "other"]),
    "sex": (group_sex, ["female", "male"]),
}

TAB_YEAR_GROUP_ORDER: List[str] = [
    "1975-2000",
    "2000-2003",
    "2004-2005",
    "2006",
    "2007",
    "2008",
    "2009",
    "2010-2011",
    "2012-2017",
]

TAB_COUNTRY_GROUP_ORDER: List[str] = [
    "TUR",
    "POL",
    "GBR-IRL-ROU;GBR",
    "AUT-DEU-CHE-LIE-AUT;SVN",
    "SWE-NOR-DNK",
    "FRA-ESP-BEL-SMR",
]

RAT_BENCH_STATE_REGIONS: Dict[str, str] = {
    # US Census Bureau's four-region breakdown.
    "Connecticut/CT": "Northeast", "Maine/ME": "Northeast", "Massachusetts/MA": "Northeast",
    "New Hampshire/NH": "Northeast", "Rhode Island/RI": "Northeast", "Vermont/VT": "Northeast",
    "New Jersey/NJ": "Northeast", "New York/NY": "Northeast", "Pennsylvania/PA": "Northeast",
    "Illinois/IL": "Midwest", "Indiana/IN": "Midwest", "Michigan/MI": "Midwest",
    "Ohio/OH": "Midwest", "Wisconsin/WI": "Midwest", "Iowa/IA": "Midwest",
    "Kansas/KS": "Midwest", "Minnesota/MN": "Midwest", "Missouri/MO": "Midwest",
    "Nebraska/NE": "Midwest", "North Dakota/ND": "Midwest", "South Dakota/SD": "Midwest",
    "Delaware/DE": "South", "Florida/FL": "South", "Georgia/GA": "South",
    "Maryland/MD": "South", "North Carolina/NC": "South", "South Carolina/SC": "South",
    "Virginia/VA": "South", "District of Columbia/DC": "South", "West Virginia/WV": "South",
    "Alabama/AL": "South", "Kentucky/KY": "South", "Mississippi/MS": "South",
    "Tennessee/TN": "South", "Arkansas/AR": "South", "Louisiana/LA": "South",
    "Oklahoma/OK": "South", "Texas/TX": "South",
    "Arizona/AZ": "West", "Colorado/CO": "West", "Idaho/ID": "West",
    "Montana/MT": "West", "Nevada/NV": "West", "New Mexico/NM": "West",
    "Utah/UT": "West", "Wyoming/WY": "West", "Alaska/AK": "West",
    "California/CA": "West", "Hawaii/HI": "West", "Oregon/OR": "West", "Washington/WA": "West",
}
RAT_BENCH_EDUCATION_GROUPS: Dict[str, str] = {
    "No schooling completed": "no_schooling",
    "Nursery school, preschool": "no_schooling",
    "N/A (less than 3 years old)": "no_schooling",
    "Grade 4": "secondary", "Grade 6": "secondary", "Grade 8": "secondary",
    "Grade 10": "secondary", "Grade 11": "secondary", "12th grade - no diploma": "secondary",
    "Regular high school diploma": "secondary", "GED or alternative credential": "secondary",
    "Some college, but less than 1 year": "some_college",
    "1 or more years of college credit, no degree": "some_college",
    "Associate's degree": "some_college",
    "Bachelor's degree": "bachelor",
    "Master's degree": "master",
    "Professional degree beyond a bachelor's degree": "master",
    "Doctorate degree": "doctorate",
}
RAT_BENCH_EDUCATION_GROUP_ORDER: List[str] = [
    "no_schooling", "secondary", "some_college", "bachelor", "master", "doctorate",
]

RAT_BENCH_BIRTH_DECADE_ORDER: List[str] = [f"{decade}s" for decade in range(1930, 2030, 10)]

ORDINAL_LABEL_ORDERS: Dict[str, Dict[str, List[str]]] = {
    "reddit": {feature: list(order) for feature, (_, order) in ORDINAL_GROUPERS.items()},
    "tab": {"year_group": list(TAB_YEAR_GROUP_ORDER)},
    "rat_bench": {
        "educational_attainment_group": list(RAT_BENCH_EDUCATION_GROUP_ORDER),
        "birth_decade": list(RAT_BENCH_BIRTH_DECADE_ORDER),
    },
}

NOMINAL_GROUPERS: Dict[str, Callable[[str], str]] = {
    "birth_city_country": group_region,
    "city_country": group_region,
    "occupation": group_occupation,
    "relationship_status": group_relationship,
}

GROUPERS: Dict[str, Callable[[str], str]] = {
    **{feature: func for feature, (func, _) in ORDINAL_GROUPERS.items()},
    **NOMINAL_GROUPERS,
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

def reddit_hardness(record: DatasetRecord) -> Optional[str]:
    return text_value(record.metadata.get("hardness"))


def reddit_question(record: DatasetRecord) -> Optional[str]:
    return text_value(record.metadata.get("question"))


def tab_country(record: DatasetRecord) -> Optional[str]:
    return text_value(record.metadata.get("country"))


def tab_year(record: DatasetRecord) -> Optional[int]:
    value = record.metadata.get("year")
    if isinstance(value, (list, tuple, set)):
        for item in value:
            number = int_value(item)
            if number is not None:
                return number
        return None
    return int_value(value)

def tab_legal_branch(record: DatasetRecord) -> Optional[str]:
    return text_value(record.metadata.get("legal_branch"))

def tab_articles(record: DatasetRecord) -> Optional[List[str]]:
    value = record.metadata.get("articles")
    if isinstance(value, (list, tuple, set)):
        articles = sorted(list(set([text_value(item) for item in value])), key=lambda x: int_value(x) if x and x.isdigit() else float('inf'))
        return ",".join([a for a in articles if a is not None])
    raise ValueError("Articles metadata is not a list.")

def tab_year_groups(record: DatasetRecord, year_groups: List[str] | None = None) -> str:
    groups = year_groups or TAB_YEAR_GROUP_ORDER
    year = record.metadata.get("year")
    bounds = [(int(g.split("-")[0]), int(g.split("-")[1])) if "-" in g else (int(g), int(g)) for g in groups]
    mapping = {c: g for g, (start, end) in zip(groups, bounds) for c in range(start, end + 1)}
    if year not in mapping:
        raise ValueError(f"Year {year} not in any defined year groups.")
    return mapping[year]


def tab_country_groups(record: DatasetRecord, region_groups: List[str] | None = None) -> str:
    groups = region_groups or TAB_COUNTRY_GROUP_ORDER
    region = record.metadata.get("country")
    mapping = {c: g for g in groups for c in g.split("-")}
    return mapping.get(region, region)


def db_bio_label(record: DatasetRecord) -> Optional[str]:
    return text_value(record.metadata.get("label"))

def db_bio_l1(record: DatasetRecord) -> Optional[str]:
    return text_value(record.metadata.get("l1"))

def db_bio_l2(record: DatasetRecord) -> Optional[str]:
    return text_value(record.metadata.get("l2"))

def db_bio_l3(record: DatasetRecord) -> Optional[str]:
    return text_value(record.metadata.get("l3"))

_RAT_BENCH_OCCUPATION_TYPE_RE = re.compile(r"TYPE:\s*(.+?),\s*DESCRIPTION:")
_RAT_BENCH_DOB_YEAR_RE = re.compile(r"(\d{4})\s*$")


def rat_bench_scenario(record: DatasetRecord) -> Optional[str]:
    return text_value(record.metadata.get("scenario"))


def rat_bench_sex(record: DatasetRecord) -> Optional[str]:
    value = text_value(record.metadata.get("profile_sex"))
    return value.lower() if value else None


def rat_bench_employment_status(record: DatasetRecord) -> Optional[str]:
    return text_value(record.metadata.get("profile_employment_status"))


def rat_bench_citizenship_status(record: DatasetRecord) -> Optional[str]:
    return text_value(record.metadata.get("profile_citizenship_status"))


def rat_bench_marital_status(record: DatasetRecord) -> Optional[str]:
    return text_value(record.metadata.get("profile_marital_status"))


def rat_bench_occupation_group(record: DatasetRecord) -> Optional[str]:
    occupation = text_value(record.metadata.get("profile_occupation"))
    if occupation is None:
        return None
    match = _RAT_BENCH_OCCUPATION_TYPE_RE.match(occupation)
    return match.group(1) if match else occupation


def rat_bench_educational_attainment_group(record: DatasetRecord) -> Optional[str]:
    label = text_value(record.metadata.get("profile_educational_attainment"))
    if label is None:
        return None
    group = RAT_BENCH_EDUCATION_GROUPS.get(label)
    if group is None:
        raise ValueError(f"Unknown RAT-Bench educational attainment label: {label}")
    return group


def rat_bench_state_region(record: DatasetRecord) -> Optional[str]:
    label = text_value(record.metadata.get("profile_state_of_residence"))
    if label is None:
        return None
    region = RAT_BENCH_STATE_REGIONS.get(label)
    if region is None:
        raise ValueError(f"Unknown RAT-Bench state label: {label}")
    return region


def rat_bench_birth_decade(record: DatasetRecord) -> Optional[str]:
    dob = text_value(record.metadata.get("profile_date_of_birth"))
    if dob is None:
        return None
    match = _RAT_BENCH_DOB_YEAR_RE.search(dob)
    if match is None:
        raise ValueError(f"Could not parse a year out of RAT-Bench date of birth: {dob}")
    decade = (int(match.group(1)) // 10) * 10
    return f"{decade}s"


def yelp_is_positive(record: DatasetRecord) -> Optional[bool]:
    stars = record.metadata.get("stars")
    if stars is None:
        return None
    if isinstance(stars, (int, float)):
        return stars >= 3
    if isinstance(stars, str):
        try:
            return float(stars) >= 3
        except ValueError:
            return None
    return None


DERIVE_REGISTRY: Dict[str, Dict[str, Callable[[DatasetRecord], Any]]] = {
    "reddit": {
        "feature": reddit_feature,
        "label": reddit_label,
        "hardness": reddit_hardness,
        "question": reddit_question,
        "feature_label": reddit_feature_label,
        "feature_label_exact": lambda r: reddit_feature_label(r, group=False),
    },
    "tab": {
        "year": tab_year,
        "country": tab_country,
        "year_group": tab_year_groups,
        "country_group": tab_country_groups,
        "legal_branch": tab_legal_branch,
        "articles": tab_articles,
    },
    "db_bio": {
        "label": db_bio_label,
        "l1": db_bio_l1,
        "l2": db_bio_l2,
        "l3": db_bio_l3,
    },
    "yelp": {
        "is_positive": yelp_is_positive,
    },
    "rat_bench": {
        "scenario": rat_bench_scenario,
        "sex": rat_bench_sex,
        "employment_status": rat_bench_employment_status,
        "citizenship_status": rat_bench_citizenship_status,
        "marital_status": rat_bench_marital_status,
        "occupation_group": rat_bench_occupation_group,
        "educational_attainment_group": rat_bench_educational_attainment_group,
        "state_region": rat_bench_state_region,
        "birth_decade": rat_bench_birth_decade,
    },
}


def get_getter(dataset: str, key: str) -> Callable[[DatasetRecord], Any]:
    if dataset not in DERIVE_REGISTRY:
        raise ValueError(f"Unknown dataset '{dataset}' for derive getters.")
    if key not in DERIVE_REGISTRY[dataset]:
        raise ValueError(f"Unknown key '{key}' for dataset '{dataset}' in derive getters.")
    return DERIVE_REGISTRY[dataset][key]


def get_ordinal_label_order(dataset: str, key: str, feature: Optional[str] = None) -> Optional[List[str]]:
    if dataset == "reddit" and key == "feature_label" and feature:
        pair = ORDINAL_GROUPERS.get(feature)
        if pair is None:
            return None
        _, order = pair
        return [f"{feature}: {label}" for label in order]
    by_dataset = ORDINAL_LABEL_ORDERS.get(dataset)
    if by_dataset is None:
        return None
    order = by_dataset.get(key)
    if order is None:
        return None
    return list(order)