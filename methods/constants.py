from dataclasses import dataclass
from typing import Dict


@dataclass
class ModelCapabilities:
    requires_dataset: bool
    uses_annotations: bool = False
    uses_scoring: bool = False
    uses_filtering: bool = False
    requires_non_uniform_explainer: bool = False


MODEL_CAPABILITIES: Dict[str, ModelCapabilities] = {
    "spacy": ModelCapabilities(requires_dataset=False),
    "presidio": ModelCapabilities(requires_dataset=False),
    "manual": ModelCapabilities(requires_dataset=True),
    "baroud": ModelCapabilities(requires_dataset=False),
    "petre": ModelCapabilities(
        requires_dataset=True,
        uses_annotations=True,
        uses_scoring=True,
        requires_non_uniform_explainer=True,
    ),
    "dpbart": ModelCapabilities(requires_dataset=False),
    "dpparaphrase": ModelCapabilities(requires_dataset=False),
    "dpprompt": ModelCapabilities(requires_dataset=False),
    "dpmlm": ModelCapabilities(
        requires_dataset=False,
        uses_filtering=True,
        uses_scoring=True,
    ),
}


def get_capabilities(model_name: str) -> ModelCapabilities:
    if model_name not in MODEL_CAPABILITIES:
        raise ValueError(f"Unknown model: {model_name}")
    return MODEL_CAPABILITIES[model_name]


def requires_dataset(model_name: str) -> bool:
    return get_capabilities(model_name).requires_dataset
