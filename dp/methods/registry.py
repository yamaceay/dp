from typing import Dict, Type
from dp.methods.anonymizer import Anonymizer
from dp.methods.constants import ModelCapabilities

from dp.methods._spacy import SpacyAnonymizer
from dp.methods._manual import ManualAnonymizer
from dp.methods._presidio import PresidioAnonymizer
from dp.methods._baroud import BaroudAnonymizer
from dp.methods._risk import RiskAnonymizer
from dp.methods._petre import PetreAnonymizer

from dp.methods._dpbart import DPBartAnonymizer
from dp.methods._dpparaphrase import DPParaphraseAnonymizer
from dp.methods._dpprompt import DPPromptAnonymizer
from dp.methods._dpmlm import DPMlmAnonymizer
from dp.methods._iter_dpmlm import IterDPMlmAnonymizer
from dp.methods._iter_petre import IterPetreAnonymizer

MODEL_REGISTRY: Dict[str, Type[Anonymizer]] = {
    "spacy": SpacyAnonymizer,
    "presidio": PresidioAnonymizer,
    "manual": ManualAnonymizer,
    "baroud": BaroudAnonymizer,
    "risk": RiskAnonymizer,
    "petre": PetreAnonymizer,
    "dpbart": DPBartAnonymizer,
    "dpparaphrase": DPParaphraseAnonymizer,
    "dpprompt": DPPromptAnonymizer,
    "dpmlm": DPMlmAnonymizer,
    "iter_dpmlm": IterDPMlmAnonymizer,
    "iter_petre": IterPetreAnonymizer,
}

MODEL_CAPABILITIES: Dict[str, ModelCapabilities] = {
    "spacy": ModelCapabilities(),
    "presidio": ModelCapabilities(),
    "manual": ModelCapabilities(
        must_use_dataset=True, 
        must_use_annotations=True,
    ),
    "baroud": ModelCapabilities(must_use_pii_selector=True),
    "risk": ModelCapabilities(
        must_use_risk_selector=True,
        must_use_scoring=True,
    ),
    "petre": ModelCapabilities(
        must_use_dataset=True, 
        can_use_k_selector=True,
        can_use_annotations=True,
        must_use_scoring=True,
    ),
    "dpbart": ModelCapabilities(can_work_token_level=False),
    "dpparaphrase": ModelCapabilities(can_work_token_level=False),
    "dpprompt": ModelCapabilities(can_work_token_level=False),
    "dpmlm": ModelCapabilities(
        can_use_dataset=True,
        can_use_pii_selector=True,
        can_use_risk_selector=True,
        can_use_k_selector=True,
        can_use_scoring=True,
    ),
    "iter_dpmlm": ModelCapabilities(
        can_use_dataset=True,
        can_use_k_selector=True,
        can_use_scoring=True,
    ),
    "iter_petre": ModelCapabilities(
        must_use_dataset=True,
        can_use_k_selector=True,
        can_use_annotations=True,
        must_use_scoring=True,
    ),
}

def get_capabilities(model_name: str) -> ModelCapabilities:
    if model_name not in MODEL_CAPABILITIES:
        raise ValueError(f"Unknown model: {model_name}")
    return MODEL_CAPABILITIES[model_name]