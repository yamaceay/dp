from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from dp.experiments.utility.base import UtilityTarget
from dp.experiments.utility.vectorizer import SelfSupervisedFeatureExtractor, FEATURE_EXTRACTOR_REGISTRY
from dp.bert import SupervisedDownstreamHead
from dp.experiments.utility.downstream import DOWNSTREAM_HEAD_REGISTRY


MODE_TO_MODELS: Dict[UtilityTarget.Mode, List[Tuple[str, str]]] = {
    UtilityTarget.Mode.BINARY: [("text", "bert_classifier"), ("text", "qwen_classifier")],
    UtilityTarget.Mode.NOMINAL: [("text", "bert_classifier"), ("text", "qwen_classifier")],
    UtilityTarget.Mode.ORDINAL: [("text", "bert_ordinal"), ("text", "qwen_ordinal")],
    UtilityTarget.Mode.CARDINAL: [("text", "bert_regressor"), ("text", "qwen_regressor")],
}

# Backward compatibility alias.
MODE_TO_MODEL = MODE_TO_MODELS


def _matches_preference(candidate: Tuple[str, str], preference: str) -> bool:
    pref = preference.strip().lower()
    if not pref:
        return False
    vectorizer_name, head_name = candidate
    return pref in vectorizer_name.lower() or pref in head_name.lower()


def resolve_model_choice(mode: UtilityTarget.Mode, preference: Optional[str] = None) -> Tuple[str, str]:
    candidates = MODE_TO_MODELS.get(mode) or []
    if not candidates:
        raise ValueError(f"no model candidates configured for mode '{mode.value}'")

    if preference:
        for cand in candidates:
            if _matches_preference(cand, preference):
                return cand
        available = ", ".join([f"{v}+{h}" for v, h in candidates])
        raise ValueError(
            f"unknown preference '{preference}' for mode '{mode.value}'. "
            f"Available candidates: {available}"
        )
    # Default to first item (BERT-first convention).
    return candidates[0]


@dataclass(frozen=True)
class UtilitySpec:
    dataset: str
    target_key: str
    target: UtilityTarget
    default_vectorizer: str
    default_head: str
    preference: Optional[str] = None

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
        v_name = (vectorizer_name or "").lower()
        h_name = (head_name or "").lower()
        if not v_name or not h_name:
            pref_vec, pref_head = resolve_model_choice(self.target.mode, self.preference)
            if not v_name:
                v_name = (self.default_vectorizer or pref_vec or "").lower()
            if not h_name:
                h_name = (self.default_head or pref_head or "").lower()
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
    "MODE_TO_MODELS",
    "MODE_TO_MODEL",
    "resolve_model_choice",
]
