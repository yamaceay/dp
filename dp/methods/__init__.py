"""Methods package re-exporting anonymizer types and registry.

This module exposes a compact API mirroring the previous single-file
implementation but organized into submodules for clarity.
"""

from dp.methods import registry
from dp.methods.anonymizer import Anonymizer, AnonymizationResult

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
from dp.methods._dpmlm_longformer import DPMlmLongformerAnonymizer

__all__ = [
    "Anonymizer",
    "AnonymizationResult",
    "ManualAnonymizer",
    "SpacyAnonymizer",
    "PresidioAnonymizer",
    "BaroudAnonymizer",
    "RiskAnonymizer",
    "PetreAnonymizer",
    "DPBartAnonymizer",
    "DPParaphraseAnonymizer",
    "DPPromptAnonymizer",
    "DPMlmAnonymizer",
    "DPMlmLongformerAnonymizer",
    "registry",
]
