from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

from dp.experiments.utility.base import UtilityTarget
from dp.experiments.utility.models import SupervisedDownstreamHead, LinearRegressor, LogisticClassifier, FeedForwardClassifier, FeedForwardRegressor
from dp.experiments.utility.vectorizer import SelfSupervisedFeatureExtractor, TfidfTextVectorizer, BERTVectorizer
from dp.experiments.utility.constants import UTILITY_TARGETS 
from dp.loaders.base import DatasetRecord

VECTORIZER_MODELS: Dict[str, SelfSupervisedFeatureExtractor] = {
    "tfidf": TfidfTextVectorizer(),
    "bert": BERTVectorizer(),
}

DOWNSTREAM_CLASSIFIERS: Dict[str, SupervisedDownstreamHead] = {
    "logistic_classifier": LogisticClassifier(),
    "feedforward_classifier": FeedForwardClassifier(),
}

DOWNSTREAM_REGRESSORS: Dict[str, SupervisedDownstreamHead] = {
    "linear_regressor": LinearRegressor(), 
    "feedforward_regressor": FeedForwardRegressor(),
}

DOWNSTREAM_MODELS: Dict[str, SupervisedDownstreamHead] = {
    **DOWNSTREAM_CLASSIFIERS,
    **DOWNSTREAM_REGRESSORS,
}

MODE_TO_MODEL: Dict[UtilityTarget.Mode, Tuple[str, str]] = {
    UtilityTarget.Mode.BINARY: ("bert", "feedforward_classifier"),
    UtilityTarget.Mode.NOMINAL: ("bert", "feedforward_classifier"),
    UtilityTarget.Mode.ORDINAL: ("bert", "feedforward_classifier"),
    UtilityTarget.Mode.CARDINAL: ("bert", "feedforward_regressor"),
}

@dataclass(frozen=True)
class UtilitySpec:
    dataset: str
    target_key: str
    target: UtilityTarget
    model_name: str

    def identifier(self) -> str:
        return f"{self.dataset}_{self.target_key}"

    def build_model(self) -> SupervisedDownstreamHead:
        prototype = DOWNSTREAM_MODELS[self.model_name]
        return prototype.setup()

def _build_registry(targets: Dict[str, Dict[str, UtilityTarget]]) -> Dict[str, UtilitySpec]:
    registry: Dict[str, UtilitySpec] = {}
    for dataset, mapping in targets.items():
        for key, target in mapping.items():
            model_name = MODE_TO_MODEL[target.mode]
            spec = UtilitySpec(dataset=dataset, target_key=key, target=target, model_name=model_name)
            registry[spec.identifier()] = spec
    return registry

UTILITY_EXPERIMENTS_REGISTRY: Dict[str, UtilitySpec] = _build_registry(UTILITY_TARGETS)