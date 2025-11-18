from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

from dp.experiments.utility.base import UtilityTarget
from dp.experiments.utility.getters import UTILITY_TARGETS
from dp.experiments.utility.vectorizer import SelfSupervisedFeatureExtractor, FEATURE_EXTRACTOR_REGISTRY
from dp.experiments.utility.downstream import SupervisedDownstreamHead, DOWNSTREAM_HEAD_REGISTRY


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
    default_vectorizer: str
    default_head: str

    def identifier(self) -> str:
        return f"{self.dataset}_{self.target_key}"

    def build_components(
        self,
        *,
        vectorizer_name: Optional[str] = None,
        vectorizer_kwargs: Optional[Dict[str, Any]] = None,
        head_name: Optional[str] = None,
        head_kwargs: Optional[Dict[str, Any]] = None,
        identifier: Optional[str] = None,
    ) -> Tuple[SelfSupervisedFeatureExtractor, SupervisedDownstreamHead]:
        v_name = (vectorizer_name or self.default_vectorizer or "").lower()
        h_name = (head_name or self.default_head or "").lower()
        if identifier:
            parts = identifier.lower().replace(" ", "").replace("/", "+").split("+")
            if len(parts) == 2:
                v_name, h_name = parts[0], parts[1]
        v_kwargs = dict(vectorizer_kwargs or {})
        h_kwargs = dict(head_kwargs or {})
        if v_name not in FEATURE_EXTRACTOR_REGISTRY:
            raise ValueError(f"unknown vectorizer '{v_name}'")
        if h_name not in DOWNSTREAM_HEAD_REGISTRY:
            raise ValueError(f"unknown head '{h_name}'")
        vectorizer = FEATURE_EXTRACTOR_REGISTRY[v_name](**v_kwargs)
        head = DOWNSTREAM_HEAD_REGISTRY[h_name](**h_kwargs)
        return vectorizer, head


def _build_registry(targets: Dict[str, Dict[str, UtilityTarget]]) -> Dict[str, UtilitySpec]:
    registry: Dict[str, UtilitySpec] = {}
    for dataset, mapping in targets.items():
        for key, target in mapping.items():
            v_name, h_name = MODE_TO_MODEL[target.mode]
            spec = UtilitySpec(
                dataset=dataset,
                target_key=key,
                target=target,
                default_vectorizer=v_name,
                default_head=h_name,
            )
            registry[spec.identifier()] = spec
    return registry


UTILITY_EXPERIMENTS_REGISTRY: Dict[str, UtilitySpec] = _build_registry(UTILITY_TARGETS)