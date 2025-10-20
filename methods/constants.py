from dataclasses import dataclass
from typing import Dict


@dataclass
class ModelCapabilities:
    must_use_dataset: bool = False
    must_use_non_uniform_explainer: bool = False
    can_use_annotations: bool = False
    can_use_scoring: bool = False
    can_use_filtering: bool = False
    supports_batch_predict: bool = False


MODEL_CAPABILITIES: Dict[str, ModelCapabilities] = {
    "spacy": ModelCapabilities(),
    "presidio": ModelCapabilities(),
    "manual": ModelCapabilities(must_use_dataset=True),
    "baroud": ModelCapabilities(supports_batch_predict=True),
    "petre": ModelCapabilities(
        must_use_dataset=True,
        must_use_non_uniform_explainer=True,
        can_use_annotations=True,
        can_use_scoring=True,
    ),
    "dpbart": ModelCapabilities(),
    "dpparaphrase": ModelCapabilities(),
    "dpprompt": ModelCapabilities(),
    "dpmlm": ModelCapabilities(
        can_use_filtering=True,
        can_use_scoring=True,
    ),
}


def get_capabilities(model_name: str) -> ModelCapabilities:
    if model_name not in MODEL_CAPABILITIES:
        raise ValueError(f"Unknown model: {model_name}")
    return MODEL_CAPABILITIES[model_name]


def requires_dataset(model_name: str) -> bool:
    return get_capabilities(model_name).must_use_dataset
