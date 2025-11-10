from typing import Any, Dict, Iterator, List, Optional, Union
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
    pii_confidences: Optional[List[Optional[float]]] = None
    risk_tolerances: Optional[List[Optional[float]]] = None
    
    def has_texts(self) -> bool:
        return self.texts is not None
    
    def has_indices(self) -> bool:
        return self.indices is not None
    
    def has_epsilons(self) -> bool:
        return self.epsilons is not None
    
    def has_ks(self) -> bool:
        return self.ks is not None
    
    def has_pii_confidences(self) -> bool:
        return self.pii_confidences is not None
    
    def has_risk_tolerances(self) -> bool:
        return self.risk_tolerances is not None
    
    def is_batch_text(self) -> bool:
        return self.has_texts() and len(self.texts) > 1
    
    def is_batch_index(self) -> bool:
        return self.has_indices() and len(self.indices) > 1
    
    def is_grid_epsilon(self) -> bool:
        return self.has_epsilons() and len(self.epsilons) > 1
    
    def is_grid_k(self) -> bool:
        return self.has_ks() and len(self.ks) > 1
    
    def is_grid_pii_confidence(self) -> bool:
        return self.has_pii_confidences() and len(self.pii_confidences) > 1
    
    def is_grid_risk_tolerance(self) -> bool:
        return self.has_risk_tolerances() and len(self.risk_tolerances) > 1


class AnonymizationBuilder:
    def __init__(self, anonymizer: 'Anonymizer', model_name: Optional[str] = None):
        self.anonymizer = anonymizer
        self.model_name = model_name
        self.capabilities = get_capabilities(model_name) if model_name else None
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
    
    def with_pii_confidences(self, values: Union[None, float, List[Optional[float]]]) -> 'AnonymizationBuilder':
        if values is None:
            self.request.pii_confidences = None
            return self
        if isinstance(values, (int, float)):
            values = [float(values)]
        normalized: List[Optional[float]] = []
        for value in values:
            if value is None:
                normalized.append(None)
            else:
                normalized.append(float(value))
        self.request.pii_confidences = normalized
        return self
    
    def with_risk_tolerances(self, values: Union[None, float, List[Optional[float]]]) -> 'AnonymizationBuilder':
        if values is None:
            self.request.risk_tolerances = None
            return self
        if isinstance(values, (int, float)):
            values = [float(values)]
        normalized: List[Optional[float]] = []
        for value in values:
            if value is None:
                normalized.append(None)
            else:
                normalized.append(float(value))
        self.request.risk_tolerances = normalized
        return self
    
    def anonymize_stream(self, progress: bool = False, **kwargs) -> Iterator[Dict[float, List[AnonymizationResult]] | Dict[int, List[AnonymizationResult]]]:
        if self.model_name is None:
            raise ValueError("Model name not set in anonymizer")
        if self.capabilities is None:
            self.capabilities = get_capabilities(self.model_name)
        if not self.capabilities.supports_streaming:
            raise ValueError(f"{self.model_name} does not support streaming")
        if self.capabilities.requires_k:
            indices, ordered_k, filtered_kwargs = self._prepare_k_inputs(kwargs)
            return self.anonymizer.anonymize_stream(
                indices=indices,
                k=ordered_k,
                progress=progress,
                **filtered_kwargs,
            )
        if self.capabilities.requires_epsilon:
            texts, ordered_eps, filtered_kwargs = self._prepare_dp_inputs(kwargs)
            return self.anonymizer.anonymize_stream(
                texts=texts,
                epsilon=ordered_eps,
                progress=progress,
                **filtered_kwargs,
            )
        return self.anonymizer.anonymize_stream(
            texts=self.request.texts or [],
            progress=progress,            
            **kwargs,
        )

    def anonymize(self, **kwargs):
        if self.model_name is None:
            raise ValueError("Model name not set in anonymizer")
        if self.capabilities is None:
            self.capabilities = get_capabilities(self.model_name)
        name = self.anonymizer.__class__.__name__
        
        if self.request.has_texts() and self.request.has_indices():
            raise ValueError("Cannot specify both texts and indices")
        
        if self.capabilities.requires_k:
            if self.request.has_texts():
                raise ValueError(f"{name} requires dataset indices, not texts")
            if not self.request.has_indices():
                raise ValueError("Must specify indices for k-anonymization methods")
            if not self.request.has_ks():
                raise ValueError("Must specify k values for k-anonymization")
            
            return self._anonymize_k_anon(**kwargs)
        
        elif self.capabilities.requires_epsilon:
            if self.request.has_indices():
                raise ValueError(f"{name} requires texts, not dataset indices")
            if not self.request.has_texts():
                raise ValueError("Must specify texts for DP methods")
            if not self.request.has_epsilons():
                raise ValueError("Must specify epsilon values for DP methods")
            
            return self._anonymize_dp(**kwargs)
        
        elif self.capabilities.is_pii_classifier:
            if self.request.has_indices():
                raise ValueError(f"{name} requires texts, not dataset indices")
            if not self.request.has_texts():
                raise ValueError("Must specify texts for PII classifiers")
            return self._anonymize_pii_classifier(**kwargs)
        
        elif self.capabilities.is_risk_masker:
            if self.request.has_indices():
                raise ValueError(f"{name} requires texts, not dataset indices")
            if not self.request.has_texts():
                raise ValueError("Must specify texts for risk maskers")
            return self._anonymize_risk_masker(**kwargs)
        
        elif self.capabilities.must_use_dataset:
            if self.request.has_texts():
                raise ValueError(f"{name} requires dataset indices, not texts")
            if not self.request.has_indices():
                raise ValueError("Must specify indices for dataset-based methods")
            
            return self._anonymize_dataset(**kwargs)
        
        else:
            if self.request.has_indices():
                raise ValueError(f"{name} requires texts, not dataset indices")
            if not self.request.has_texts():
                raise ValueError("Must specify texts for text anonymization")

            return self._anonymize_texts(**kwargs)

    def _anonymize_texts(self, progress: bool = False, **kwargs):
        texts, record_names, base_kwargs = self._prepare_text_inputs(kwargs)
        return self._run_plain_text(texts, record_names, base_kwargs, progress)

    def _anonymize_pii_classifier(self, progress: bool = False, **kwargs):
        return self._anonymize_with_param(
            param_name="pii_confidence",
            values=self.request.pii_confidences,
            setter_name="set_pii_confidence",
            progress=progress,
            **kwargs,
        )

    def _anonymize_risk_masker(self, progress: bool = False, **kwargs):
        return self._anonymize_with_param(
            param_name="risk_tolerance",
            values=self.request.risk_tolerances,
            setter_name="set_risk_tolerance",
            progress=progress,
            **kwargs,
        )
    
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
        record_names = kwargs.get("record_names")
        filtered_kwargs = {
            key: value
            for key, value in kwargs.items()
            if key not in {"epsilon", "record_names"}
        }
        if record_names is not None:
            filtered_kwargs["record_names_iter"] = iter(record_names)
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

    def _prepare_text_inputs(self, kwargs):
        texts = list(self.request.texts or [])
        if not texts:
            raise ValueError("No texts provided for text anonymization")
        record_names = kwargs.pop("record_names", None)
        base_kwargs = dict(kwargs)
        return texts, record_names, base_kwargs

    def _run_plain_text(
        self,
        texts: List[str],
        record_names: Optional[List[str]],
        base_kwargs: Dict[str, Any],
        progress: bool,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> List[AnonymizationResult]:
        iterator = texts
        if progress:
            iterator = tqdm(texts, desc="Anonymizing texts")
        names_list = list(record_names) if record_names is not None else None
        results: List[AnonymizationResult] = []
        for idx, text in enumerate(iterator):
            per_kwargs = dict(base_kwargs)
            if names_list is not None and idx < len(names_list):
                per_kwargs["record_name"] = names_list[idx]
            result = self.anonymizer.anonymize(text=text, **per_kwargs)
            if metadata:
                payload = dict(result.metadata or {})
                payload.update(metadata)
                result.metadata = payload
            results.append(result)
        return results

    def _anonymize_with_param(
        self,
        param_name: str,
        values: Optional[List[Optional[float]]],
        setter_name: str,
        *,
        progress: bool = False,
        **kwargs,
    ):
        texts, record_names, base_kwargs = self._prepare_text_inputs(kwargs)
        ordered_values = self._normalize_parameter_values(values)
        setter = getattr(self.anonymizer, setter_name, None)
        if setter is None and any(value is not None for value in ordered_values):
            raise ValueError(f"{self.anonymizer.__class__.__name__} does not support '{param_name}' overrides")
        def run_for(value: Optional[float]):
            if value is not None and setter is not None:
                setter(value)
            metadata = {param_name: value} if value is not None else None
            return self._run_plain_text(texts, record_names, base_kwargs, progress, metadata)
        if len(ordered_values) == 1:
            return run_for(ordered_values[0])
        aggregated: Dict[Optional[float], List[AnonymizationResult]] = {}
        for value in ordered_values:
            aggregated[value] = run_for(value)
        return aggregated

    def _normalize_parameter_values(self, values: Optional[List[Optional[float]]]) -> List[Optional[float]]:
        if not values:
            return [None]
        ordered: List[Optional[float]] = []
        seen: set = set()
        for value in values:
            key = value
            if key in seen:
                continue
            seen.add(key)
            ordered.append(value)
        return ordered
