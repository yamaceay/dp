from typing import Dict

from dp.experiments import Experiment
from dp.experiments.utility.base import TextUtilityExperiment, UtilityTarget, DownstreamModel
from dp.experiments.privacy_annotations import TextPrivacyExperiment
from dp.experiments.semantic_divergence import TextDivergenceExperiment

UTILITY_EXPERIMENTS_REGISTRY: Dict[str, TextUtilityExperiment] = {
    ...
}

PRIVACY_EXPERIMENTS_REGISTRY: Dict[str, TextPrivacyExperiment] = {
    "annotation_privacy": TextPrivacyExperiment,
}

DIVERGENCE_EXPERIMENTS_REGISTRY: Dict[str, TextDivergenceExperiment] = {
    "bertscore_divergence": TextDivergenceExperiment,
}

EXPERIMENTS_REGISTRY: Dict[str, Experiment] = {
    **UTILITY_EXPERIMENTS_REGISTRY,
    **PRIVACY_EXPERIMENTS_REGISTRY,
    **DIVERGENCE_EXPERIMENTS_REGISTRY,
}

is_privacy_experiment = lambda name: name in PRIVACY_EXPERIMENTS_REGISTRY
is_divergence_experiment = lambda name: name in DIVERGENCE_EXPERIMENTS_REGISTRY
is_utility_experiment = lambda name: name in UTILITY_EXPERIMENTS_REGISTRY

def get_reddit_feature(info: Dict[str, Dict]) -> str:
    return info.get("metadata", {}).get("feature", "")

def get_reddit_label(info: Dict[str, Dict]) -> str:
    return info.get("metadata", {}).get("personality", {}).get("label", "")

def get_reddit_age(info: Dict[str, Dict]) -> int:
    return info.get("metadata", {}).get("personality", {}).get("age", 0)

def get_reddit_age_group(info: Dict[str, Dict]) -> str:
    age = get_reddit_age(info)
    age_grouped = ""
    if age < 18:
        age_grouped = "under_18"
    elif 18 <= age < 25:
        age_grouped = "18_24"
    elif 25 <= age < 35:
        age_grouped = "25_34"
    elif 35 <= age < 45:
        age_grouped = "35_44"
    elif 45 <= age < 55:
        age_grouped = "45_54"
    elif 55 <= age < 65:
        age_grouped = "55_64"
    else:
        age_grouped = "65_plus"
    return age_grouped

def get_reddit_sex(info: Dict[str, Dict]) -> str:
    return info.get("metadata", {}).get("personality", {}).get("sex", "")

def get_reddit_income_level(info: Dict[str, Dict]) -> str:
    return info.get("metadata", {}).get("personality", {}).get("income_level", "")

UTILITY_TARGETS_REGISTRY: Dict[str, UtilityTarget] = {
    "reddit_feature": UtilityTarget(
        source="reddit",
        mode="nominal",
        getter=get_reddit_feature,
    ),
    "reddit_label": UtilityTarget(
        source="reddit",
        mode="nominal",
        getter=get_reddit_label,
    ),
    "reddit_age": UtilityTarget(
        source="reddit",
        mode="cardinal",
        getter=get_reddit_age,
    ),
    "reddit_age_group": UtilityTarget(
        source="reddit",
        mode="nominal",
        getter=get_reddit_age_group,
    ),
    "reddit_sex": UtilityTarget(
        source="reddit",
        mode="binary",
        getter=get_reddit_sex,
    ),
    "reddit_income_level": UtilityTarget(
        source="reddit",
        mode="ordinal",
        getter=get_reddit_income_level,
    ),
}

class LogisticRegressionModel(DownstreamModel):
    def __init__(self, **kwargs):
        super().__init__(model_name="logistic_regression", **kwargs)

class LinearRegressionModel(DownstreamModel):
    def __init__(self, **kwargs):
        super().__init__(model_name="linear_regression", **kwargs)

DOWNSTREAM_MODELS_REGISTRY: Dict[str, DownstreamModel] = {
    "logistic_regression": LogisticRegressionModel,
    "linear_regression": LinearRegressionModel,
}