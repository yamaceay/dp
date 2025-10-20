from typing import Dict, Type
from dp.methods.anonymizer import Anonymizer

from dp.methods.simple import SimpleAnonymizer
from dp.methods.simple._spacy import SpacyAnonymizer
from dp.methods.simple._manual import ManualAnonymizer
from dp.methods.simple._presidio import PresidioAnonymizer
from dp.methods.simple._baroud import BaroudAnonymizer

from dp.methods.k_anon import KAnonymizer
from dp.methods.k_anon._petre import PetreAnonymizer

from dp.methods.dp import DPAnonymizer
from dp.methods.dp._dpbart import DPBartAnonymizer
from dp.methods.dp._dpparaphrase import DPParaphraseAnonymizer
from dp.methods.dp._dpprompt import DPPromptAnonymizer
from dp.methods.dp._dpmlm import DPMlmAnonymizer

SIMPLE_MODEL_REGISTRY: Dict[str, Type[SimpleAnonymizer]] = {
    "spacy": SpacyAnonymizer,
    "presidio": PresidioAnonymizer,
    "manual": ManualAnonymizer,
    "baroud": BaroudAnonymizer,
}

K_ANON_MODEL_REGISTRY: Dict[str, Type[KAnonymizer]] = {
    "petre": PetreAnonymizer,
}

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