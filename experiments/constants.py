from typing import Dict

from dp.experiments import Experiment
from dp.experiments.utility.base import TextUtilityExperiment
from dp.experiments.utility.db_bio_label import DBBioLabelExperiment
from dp.experiments.utility.tab_country_year import TabMetadataExperiment
from dp.experiments.utility.trustpilot_stars_category import TrustpilotStarsExperiment
from dp.experiments.utility.reddit_feature import RedditUtilityExperiment
from dp.experiments.privacy_annotations import TextPrivacyExperiment
from dp.experiments.semantic_divergence import TextDivergenceExperiment

UTILITY_EXPERIMENTS_REGISTRY: Dict[str, TextUtilityExperiment] = {
    "reddit_feature": RedditUtilityExperiment,
    "db_bio_label": DBBioLabelExperiment,
    "tab_country_year": TabMetadataExperiment,
    "trustpilot_stars_category": TrustpilotStarsExperiment,
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
