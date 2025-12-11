from dataclasses import dataclass
from typing import Dict, List

PII_CLASSIFIER_MODEL_LIST: List[str] = [
    "spacy",
    "baroud",
]

RISK_MASKER_MODEL_LIST: List[str] = [
    "risk",
]

K_ANON_MODEL_LIST = [
    "petre",
    "kdpmlm",
]

DP_MODEL_LIST = [
    "dpbart",
   "dpparaphrase",
   "dpprompt",
   "dpmlm",
]

@dataclass
class ModelCapabilities:
    must_use_dataset: bool = False
    requires_epsilon: bool = False
    requires_k: bool = False
    is_pii_classifier: bool = False
    is_risk_masker: bool = False
    must_use_non_uniform_explainer: bool = False
    can_use_annotations: bool = False
    can_use_scoring: bool = False
    can_use_filtering: bool = False
    supports_batch_predict: bool = False
    supports_streaming: bool = False


MODEL_CAPABILITIES: Dict[str, ModelCapabilities] = {
    "spacy": ModelCapabilities(is_pii_classifier=True),
    "presidio": ModelCapabilities(),
    "manual": ModelCapabilities(must_use_dataset=True),
    "baroud": ModelCapabilities(
        is_pii_classifier=True,
        supports_batch_predict=True,
        supports_streaming=True,
    ),
    "risk": ModelCapabilities(
        is_risk_masker=True,
        must_use_non_uniform_explainer=True,
        can_use_scoring=True,
        supports_streaming=True,
    ),
    "petre": ModelCapabilities(
        must_use_dataset=True,
        requires_k=True,
        must_use_non_uniform_explainer=True,
        can_use_annotations=True,
        can_use_scoring=True,
        supports_streaming=True,
    ),
    "kdpmlm": ModelCapabilities(
        must_use_dataset=True,
        requires_k=True,
        must_use_non_uniform_explainer=True,
        can_use_annotations=True,
        can_use_scoring=True,
        supports_streaming=True,
    ),
    "dpbart": ModelCapabilities(requires_epsilon=True, supports_streaming=True),
    "dpparaphrase": ModelCapabilities(requires_epsilon=True, supports_streaming=True),
    "dpprompt": ModelCapabilities(requires_epsilon=True, supports_streaming=True),
    "dpmlm": ModelCapabilities(
        requires_epsilon=True,
        can_use_filtering=True,
        can_use_scoring=True,
        supports_streaming=True,
    ),
}


def get_capabilities(model_name: str) -> ModelCapabilities:
    if model_name not in MODEL_CAPABILITIES:
        raise ValueError(f"Unknown model: {model_name}")
    return MODEL_CAPABILITIES[model_name]


def must_use_dataset(model_name: str) -> bool:
    return get_capabilities(model_name).must_use_dataset
