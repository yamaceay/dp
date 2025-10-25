from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

from dp.loaders.base import DatasetRecord
from dp.experiments.utility.base import UtilityTarget
from dp.experiments.utility.downstream import DOWNSTREAM_HEAD_REGISTRY
from dp.experiments.utility.vectorizer import FEATURE_EXTRACTOR_REGISTRY
from dp.experiments.utility.constants import UTILITY_TARGETS 

class UtilityModel:
    def __init__(self) -> None:
        self._feature_extractor: Optional[SelfSupervisedFeatureExtractor] = None
        self._downstream_model: Optional[SupervisedDownstreamHead] = None

    def with_feature_extractor(self, extractor: SelfSupervisedFeatureExtractor) -> 'UtilityModel':
        self._feature_extractor = extractor
        return self

    def with_downstream_model(self, model: SupervisedDownstreamHead) -> 'UtilityModel':
        self._downstream_model = model
        return self
    
    def fit(self, texts: Sequence[str], labels: Sequence[Any]) -> None:
        if self._feature_extractor is None:
            raise ValueError("Feature extractor is not set")
        if self._downstream_model is None:
            raise ValueError("Downstream model is not set")
        self._feature_extractor.fit(texts)
        features = self._feature_extractor.transform(texts)
        self._downstream_model.fit(features, labels)

    def predict(self, texts: Sequence[str]) -> Any:
        if self._feature_extractor is None:
            raise ValueError("Feature extractor is not set")
        if self._downstream_model is None:
            raise ValueError("Downstream model is not set")
        features = self._feature_extractor.transform(texts)
        return self._downstream_model.predict(features)
    
    def evaluate(self, texts: Sequence[str], labels: Sequence[Any]) -> Dict[str, float]:
        features = self.predict(texts)
        return self._downstream_model.evaluate(features, labels)

    def identifier(self) -> str:
        feat_name = type(self._feature_extractor).__name__ if self._feature_extractor else "None"
        model_name = type(self._downstream_model).__name__ if self._downstream_model else "None"
        return f"{feat_name}_{model_name}"

@dataclass(frozen=True)
class UtilitySpec:
    dataset: str
    target_key: str
    target: UtilityTarget
    model_name: str

    def identifier(self) -> str:
        return f"{self.dataset}_{self.target_key}"

def _build_registry(targets: Dict[str, Dict[str, UtilityTarget]]) -> Dict[str, UtilitySpec]:
    registry: Dict[str, UtilitySpec] = {}
    for dataset, mapping in targets.items():
        for key, target in mapping.items():
            vectorizer_name, head_name = MODE_TO_MODEL[target.mode]
            if vectorizer_name not in FEATURE_EXTRACTOR_REGISTRY:
                raise ValueError(f"unknown feature extractor: {vectorizer_name}")
            if head_name not in DOWNSTREAM_HEAD_REGISTRY:
                raise ValueError(f"unknown downstream head: {head_name}")
            spec = (
                UtilitySpec(dataset=dataset, target_key=key, target=target)
                .with_feature_extractor(FEATURE_EXTRACTOR_REGISTRY[vectorizer_name])
                .with_downstream_model(DOWNSTREAM_HEAD_REGISTRY[head_name])
            )
            registry[spec.identifier()] = spec
    return registry

UTILITY_EXPERIMENTS_REGISTRY: Dict[str, UtilitySpec] = _build_registry(UTILITY_TARGETS)