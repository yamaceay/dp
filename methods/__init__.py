"""Methods package re-exporting anonymizer types and registry.

This module exposes a compact API mirroring the previous single-file
implementation but organized into submodules for clarity.
"""

from dp.methods import registry
from dp.methods.anonymizer import Anonymizer, AnonymizationResult

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

__all__ = [
    "Anonymizer",
    "AnonymizationResult",
    "SimpleAnonymizer",
    "ManualAnonymizer",
    "SpacyAnonymizer",
    "PresidioAnonymizer",
    "BaroudAnonymizer",
    "KAnonymizer",
    "PetreAnonymizer",
    "DPAnonymizer",
    "DPBartAnonymizer",
    "DPParaphraseAnonymizer",
    "DPPromptAnonymizer",
    "DPMlmAnonymizer",
    "registry",
]
