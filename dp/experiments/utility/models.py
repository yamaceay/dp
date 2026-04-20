from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

from dp.experiments.utility.base import UtilityTarget
from dp.experiments.utility.vectorizer import SelfSupervisedFeatureExtractor, FEATURE_EXTRACTOR_REGISTRY
from dp.bert import SupervisedDownstreamHead
from dp.experiments.utility.downstream import DOWNSTREAM_HEAD_REGISTRY


MODE_TO_MODEL: Dict[UtilityTarget.Mode, Tuple[str, str]] = {
    UtilityTarget.Mode.BINARY: ("text", "bert_classifier"),
    UtilityTarget.Mode.NOMINAL: ("text", "bert_classifier"),
    UtilityTarget.Mode.ORDINAL: ("text", "bert_ordinal"),
    UtilityTarget.Mode.CARDINAL: ("text", "bert_regressor"),
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


__all__ = [
    "UtilitySpec",
    "MODE_TO_MODEL",
]
