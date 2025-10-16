from typing import Dict, Type
from .anonymizer import Anonymizer

from .simple import SimpleAnonymizer
from .simple._spacy import SpacyAnonymizer
from .simple._presidio import PresidioAnonymizer
from .simple._baroud import BaroudAnonymizer

from .k_anon import KAnonymizer
from .k_anon._petre import PetreAnonymizer

from .dp import DPAnonymizer
from .dp._dpbart import DPBartAnonymizer
from .dp._dpparaphrase import DPParaphraseAnonymizer
from .dp._dpprompt import DPPromptAnonymizer
from .dp._dpmlm import DPMlmAnonymizer

SIMPLE_MODEL_REGISTRY: Dict[str, Type[SimpleAnonymizer]] = {
    "spacy": SpacyAnonymizer,
    "presidio": PresidioAnonymizer,
    "baroud": BaroudAnonymizer,
}

K_ANON_MODEL_REGISTRY: Dict[str, Type[KAnonymizer]] = {"petre": PetreAnonymizer}

DP_MODEL_REGISTRY: Dict[str, Type[DPAnonymizer]] = {
    "dpbart": DPBartAnonymizer,
    "dpparaphrase": DPParaphraseAnonymizer,
    "dpprompt": DPPromptAnonymizer,
    "dpmlm": DPMlmAnonymizer,
}

MODEL_REGISTRY: Dict[str, Type[Anonymizer]] = {
    **SIMPLE_MODEL_REGISTRY,
    **K_ANON_MODEL_REGISTRY,
    **DP_MODEL_REGISTRY,
}


def is_simple(AnonymizerClass: Type[Anonymizer]) -> bool:
    return AnonymizerClass in SIMPLE_MODEL_REGISTRY.values()


def is_k_anonymizer(AnonymizerClass: Type[Anonymizer]) -> bool:
    return AnonymizerClass in K_ANON_MODEL_REGISTRY.values()


def is_dp_anonymizer(AnonymizerClass: Type[Anonymizer]) -> bool:
    return AnonymizerClass in DP_MODEL_REGISTRY.values()