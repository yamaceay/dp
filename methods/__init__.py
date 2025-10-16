"""Methods package re-exporting anonymizer types and registry.

This module exposes a compact API mirroring the previous single-file
implementation but organized into submodules for clarity.
"""

from . import registry
from .anonymizer import Anonymizer, AnonymizationResult

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

__all__ = [
    "Anonymizer",
    "AnonymizationResult",
    "SimpleAnonymizer",
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
