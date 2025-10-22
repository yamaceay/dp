from typing import Optional, List, Union
from abc import ABC, abstractmethod
from dataclasses import dataclass
from tqdm import tqdm

from dp.methods.constants import get_capabilities

@dataclass
class AnonymizationResult:
    text: str
    spans: Optional[List] = None
    metadata: Optional[dict] = None


class Anonymizer(ABC):
    def __init__(self, *args, **kwargs):
        self._init_args = args
        self._init_kwargs = kwargs
        self._model_name = kwargs.get('model', None)
        print(f"Initialized {self.__class__.__name__} with args: {args}, kwargs: {kwargs}")

    def builder(self):
        return AnonymizationBuilder(self, self._model_name)

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

    @abstractmethod
    def add_dataset_records(self, dataset_records):
        """Add dataset records to the anonymizer for use in anonymization.

        Implementations should override this method to handle dataset records
        appropriately.
        """
        raise NotImplementedError()

@dataclass
class AnonymizationRequest:
    texts: Optional[List[str]] = None
    indices: Optional[List[int]] = None
    epsilons: Optional[List[float]] = None
    ks: Optional[List[int]] = None
    
    def has_texts(self) -> bool:
        return self.texts is not None
    
    def has_indices(self) -> bool:
        return self.indices is not None
    
    def has_epsilons(self) -> bool:
        return self.epsilons is not None
    
    def has_ks(self) -> bool:
        return self.ks is not None
    
    def is_batch_text(self) -> bool:
        return self.has_texts() and len(self.texts) > 1
    
    def is_batch_index(self) -> bool:
        return self.has_indices() and len(self.indices) > 1
    
    def is_grid_epsilon(self) -> bool:
        return self.has_epsilons() and len(self.epsilons) > 1
    
    def is_grid_k(self) -> bool:
        return self.has_ks() and len(self.ks) > 1


class AnonymizationBuilder:
    def __init__(self, anonymizer: 'Anonymizer', model_name: Optional[str] = None):
        self.anonymizer = anonymizer
        self.model_name = model_name
        self.request = AnonymizationRequest()
    
    def with_texts(self, texts: Union[str, List[str]]) -> 'AnonymizationBuilder':
        if isinstance(texts, str):
            texts = [texts]
        self.request.texts = texts
        return self
    
    def with_indices(self, indices: Union[int, List[int]]) -> 'AnonymizationBuilder':
        if isinstance(indices, int):
            indices = [indices]
        self.request.indices = indices
        return self
    
    def with_epsilons(self, epsilons: Union[float, List[float]]) -> 'AnonymizationBuilder':
        if isinstance(epsilons, (int, float)):
            epsilons = [float(epsilons)]
        self.request.epsilons = [float(e) for e in epsilons]
        return self
    
    def with_ks(self, ks: Union[int, List[int]]) -> 'AnonymizationBuilder':
        if isinstance(ks, int):
            ks = [ks]
        self.request.ks = ks
        return self
    
    def anonymize(self, **kwargs):
        if self.model_name is None:
            raise ValueError("Model name not set in anonymizer")
        
        capabilities = get_capabilities(self.model_name)
        name = self.anonymizer.__class__.__name__
        
        if self.request.has_texts() and self.request.has_indices():
            raise ValueError("Cannot specify both texts and indices")
        
        if capabilities.requires_k:
            if self.request.has_texts():
                raise ValueError(f"{name} requires dataset indices, not texts")
            if not self.request.has_indices():
                raise ValueError("Must specify indices for k-anonymization methods")
            if not self.request.has_ks():
                raise ValueError("Must specify k values for k-anonymization")
            
            return self._anonymize_k_anon(**kwargs)
        
        elif capabilities.requires_epsilon:
            if self.request.has_indices():
                raise ValueError(f"{name} requires texts, not dataset indices")
            if not self.request.has_texts():
                raise ValueError("Must specify texts for DP methods")
            if not self.request.has_epsilons():
                raise ValueError("Must specify epsilon values for DP methods")
            
            return self._anonymize_dp(**kwargs)
        
        elif capabilities.must_use_dataset:
            if self.request.has_texts():
                raise ValueError(f"{name} requires dataset indices, not texts")
            if not self.request.has_indices():
                raise ValueError("Must specify indices for dataset-based methods")
            
            return self._anonymize_dataset(**kwargs)
        
        else:
            if self.request.has_indices():
                raise ValueError(f"{name} requires texts, not dataset indices")
            if not self.request.has_texts():
                raise ValueError("Must specify texts for simple methods")
            
            return self._anonymize_simple(**kwargs)

    def _anonymize_simple(self, progress: bool = False, **kwargs):
        results = []
        texts_iter = self.request.texts
        if progress:
            texts_iter = tqdm(texts_iter, desc="Anonymizing texts")
        for text in texts_iter:
            result = self.anonymizer.anonymize(text=text, **kwargs)
            results.append(result)
        return results
    
    def _anonymize_dp(self, progress: bool = False, **kwargs):
        filtered_kwargs = {k: v for k, v in kwargs.items() if k != 'epsilon'}
        results = []
        texts_iter = self.request.texts
        if progress:
            texts_iter = tqdm(texts_iter, desc="Anonymizing texts with DP")
        for text in texts_iter:
            text_results = self.anonymizer.grid_anonymize(
                text=text, 
                epsilon=self.request.epsilons, 
                **filtered_kwargs
            )
            results.append(text_results)
        return results
    
    def _anonymize_k_anon(self, progress: bool = False, **kwargs):
        filtered_kwargs = {k: v for k, v in kwargs.items() if k != 'k'}
        results = []
        indices_iter = self.request.indices
        for idx in indices_iter:
            idx_results = self.anonymizer.grid_anonymize_from_dataset(
                idx=idx,
                k=self.request.ks,
                progress=progress,
                **filtered_kwargs
            )
            results.append(idx_results)
        return results

    def _anonymize_dataset(self, progress: bool = False, **kwargs):
        results = []
        indices_iter = self.request.indices
        if progress:
            indices_iter = tqdm(indices_iter, desc="Anonymizing dataset")
        for idx in indices_iter:
            result = self.anonymizer.anonymize_from_dataset(idx=idx, **kwargs)
            results.append(result)
        return results
