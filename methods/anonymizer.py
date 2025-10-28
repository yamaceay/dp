from typing import Dict, Iterator, List, Optional, Union
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
        raise NotImplementedError()
    
    @abstractmethod
    def anonymize_from_dataset(self, idx: int, *args, **kwargs) -> AnonymizationResult:
        raise NotImplementedError()

    @abstractmethod
    def add_dataset_records(self, dataset_records):
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
    
    def anonymize_stream(self, progress: bool = False, **kwargs) -> Iterator[Dict[float, List[AnonymizationResult]] | Dict[int, List[AnonymizationResult]]]:
        if self.model_name is None:
            raise ValueError("Model name not set in anonymizer")
        capabilities = get_capabilities(self.model_name)
        if not capabilities.supports_streaming:
            raise ValueError(f"{self.model_name} does not support streaming")
        if capabilities.requires_k:
            indices, ordered_k, filtered_kwargs = self._prepare_k_inputs(kwargs)
            return self.anonymizer.anonymize_stream(
                indices=indices,
                k=ordered_k,
                progress=progress,
                **filtered_kwargs,
            )
        if capabilities.requires_epsilon:
            texts, ordered_eps, filtered_kwargs = self._prepare_dp_inputs(kwargs)
            return self.anonymizer.anonymize_stream(
                texts=texts,
                epsilon=ordered_eps,
                progress=progress,
                **filtered_kwargs,
            )
        raise ValueError("Streaming not supported for this model")

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
        texts, ordered_eps, filtered_kwargs = self._prepare_dp_inputs(kwargs)
        stream = self.anonymizer.stream_batch_anonymize(
            texts=texts,
            epsilon=ordered_eps,
            progress=progress,
            **filtered_kwargs,
        )
        if len(ordered_eps) > 1:
            aggregated: Dict[float, List[List[AnonymizationResult]]] = {value: [] for value in ordered_eps}
            for per_text in stream:
                for eps_value, per_results in per_text.items():
                    aggregated.setdefault(eps_value, []).append(per_results)
            return aggregated
        single_eps = ordered_eps[0]
        results: List[List[AnonymizationResult]] = []
        for per_text in stream:
            results.append(per_text[single_eps])
        return results
    
    def _anonymize_k_anon(self, progress: bool = False, **kwargs):
        indices, ordered_k, filtered_kwargs = self._prepare_k_inputs(kwargs)
        stream = self.anonymizer.stream_batch_anonymize_from_dataset(
            indices=indices,
            k=ordered_k,
            progress=progress,
            **filtered_kwargs,
        )
        if len(ordered_k) > 1:
            aggregated: Dict[int, List[List[AnonymizationResult]]] = {value: [] for value in ordered_k}
            for per_idx in stream:
                for k_value, per_results in per_idx.items():
                    aggregated.setdefault(k_value, []).append(per_results)
            return aggregated
        single_k = ordered_k[0]
        results: List[List[AnonymizationResult]] = []
        for per_idx in stream:
            results.append(per_idx[single_k])
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

    def _prepare_dp_inputs(self, kwargs):
        texts = list(self.request.texts or [])
        epsilons = list(self.request.epsilons or [])
        if not texts:
            raise ValueError("No texts provided for DP anonymization")
        if not epsilons:
            raise ValueError("No epsilon values provided for DP anonymization")
        ordered_eps = [float(e) for e in dict.fromkeys(epsilons)]
        filtered_kwargs = {key: value for key, value in kwargs.items() if key != "epsilon"}
        return texts, ordered_eps, filtered_kwargs

    def _prepare_k_inputs(self, kwargs):
        indices = list(self.request.indices or [])
        ks = list(self.request.ks or [])
        if not indices:
            raise ValueError("No indices provided for k-anonymization")
        if not ks:
            raise ValueError("No k values provided for k-anonymization")
        ordered_k = [int(k_value) for k_value in dict.fromkeys(ks)]
        filtered_kwargs = {key: value for key, value in kwargs.items() if key != "k"}
        return indices, ordered_k, filtered_kwargs
