from typing import Optional, List
from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class AnonymizationResult:
    text: str
    spans: Optional[List] = None
    metadata: Optional[dict] = None


class Anonymizer(ABC):
    def __init__(self, *args, **kwargs):
        # lightweight base init for debugging; concrete implementations may extend
        self._init_args = args
        self._init_kwargs = kwargs
        print(f"Initialized {self.__class__.__name__} with args: {args}, kwargs: {kwargs}")

    @abstractmethod
    def anonymize(self, text: str, *args, **kwargs) -> AnonymizationResult:
        """Anonymize the provided text and return an AnonymizationResult.

        Implementations should override this method and may accept extra
        parameters (k, epsilon, etc.) specific to the algorithm.
        """
        raise NotImplementedError()
    
    @abstractmethod
    def anonymize_from_dataset(self, idx: int, *args, **kwargs) -> AnonymizationResult:
        """Anonymize the text at the specified index in the dataset and return an AnonymizationResult.

        Implementations should override this method and may accept extra
        parameters (k, epsilon, etc.) specific to the algorithm.
        """
        raise NotImplementedError()
