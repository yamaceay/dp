from typing import Any, Dict, Iterator, List, Optional, Iterable, Union, TYPE_CHECKING, Tuple
from abc import ABC
from dataclasses import dataclass
from tqdm import tqdm

from dp.methods.constants import Buckets, BucketDict
from dp.loaders.base import DatasetRecord, TextAnnotation, TextAnnotations
from dp.utils.explainer.base import TokenExplainer
from dp.utils.selector.base import AnonymizerUnit
from dp.utils.splitter import TextSplitter

if TYPE_CHECKING:
    from dp.utils.output import OutputHandler
    import torch

@dataclass
class AnonymizationResult:
    text: str
    spans: Optional[List] = None
    annotations: Optional[TextAnnotations] = None
    metadata: Optional[dict] = None

class Anonymizer(ABC):
    def __init__(self, model: str, *args, **kwargs):
        from dp.utils.output import PrintOutputHandler
        from dp.methods.registry import get_capabilities
        self._init_args = args
        self._init_kwargs = kwargs
        self._model_name = model
        self.capabilities = get_capabilities(model)

        self._output_handler: 'OutputHandler' = PrintOutputHandler()

        self._dataset_records: Optional[Iterable[DatasetRecord]] = None
        self._explainer: Optional[TokenExplainer] = None
        self._selector: Optional[AnonymizerUnit] = None
        self.device = self._resolve_device(kwargs.get("device"))

        self._starting_spans_by_uid: Dict[str, List[Tuple[int, int]]] = {}
        self._starting_annotations_name: Optional[str] = None

        print(f"Initialized {self.__class__.__name__} with args: {args}, kwargs: {kwargs}")

    def builder(self):
        return AnonymizationBuilder(self, self._model_name)

    def anonymize_any_text(self, text: str, *args, buckets: Buckets = [], **kwargs) -> List[Tuple[BucketDict, AnonymizationResult]]:
        raise NotImplementedError()
    
    def anonymize_from_dataset(self, idx: int, *args, buckets: Buckets = [], **kwargs) -> List[Tuple[BucketDict, AnonymizationResult]]:
        raise NotImplementedError()

    def anonymize(self, text_or_idx: Union[str, int], *args, buckets: Buckets = [], **kwargs) -> List[Tuple[BucketDict, AnonymizationResult]]:
        if isinstance(text_or_idx, str):
            return self.anonymize_any_text(text_or_idx, *args, buckets=buckets, **kwargs)
        return self.anonymize_from_dataset(text_or_idx, *args, buckets=buckets, **kwargs)

    def pre_stream_anonymize(self, texts_or_indices: Union[List[str], List[int]], *args, **kwargs) -> None:
        pass

    def stream_anonymize(self, texts_or_indices: Union[List[str], List[int]], *args, buckets: Buckets = [], **kwargs) -> Iterator[List[Tuple[BucketDict, AnonymizationResult]]]:
        for text in tqdm(texts_or_indices, desc="Anonymizing texts", unit="text", total=len(texts_or_indices)):
            yield self.anonymize(text, *args, buckets=buckets, **kwargs)

    def set_dataset_records(self, dataset_records: Iterable[DatasetRecord]) -> None:
        self._dataset_records = dataset_records

    def set_explainer(self, explainer: TokenExplainer) -> None:
        self._explainer = explainer

    def set_selector(self, selector: AnonymizerUnit) -> None:
        self._selector = selector

    def set_output_handler(self, output_handler: 'OutputHandler') -> None:
        self._output_handler = output_handler

    def set_splitter(self, splitter: TextSplitter) -> None:
        self._splitter = splitter

    def set_annotations(self, annotations: Dict[str, List[Any]], name: Optional[str] = None) -> None:
        spans_by_uid: Dict[str, List[Tuple[int, int]]] = {}
        for uid, raw_items in (annotations or {}).items():
            if uid is None:
                continue
            uid_str = str(uid)
            spans: List[Tuple[int, int]] = []
            if not isinstance(raw_items, list):
                raise ValueError("annotations values must be lists")
            for item in raw_items:
                if isinstance(item, TextAnnotation):
                    spans.append((int(item.start), int(item.end)))
                elif isinstance(item, dict) and "start" in item and "end" in item:
                    spans.append((int(item["start"]), int(item["end"])))
                elif isinstance(item, (list, tuple)) and len(item) >= 2:
                    spans.append((int(item[0]), int(item[1])))
            spans_by_uid[uid_str] = spans
        self._starting_spans_by_uid = spans_by_uid
        self._starting_annotations_name = name

    def _starting_indices_for_uid(self, uid: Optional[str], offsets: List[Tuple[int, int]]) -> List[int]:
        if uid is None:
            return []
        spans = self._starting_spans_by_uid.get(str(uid), [])
        if not spans or not offsets:
            return []
        indices: List[int] = []
        for idx, (ts, te) in enumerate(offsets):
            for ss, se in spans:
                if te <= ss or ts >= se:
                    continue
                indices.append(idx)
                break
        return indices

    def _resolve_device(self, device: Optional[Union[str, int, 'torch.device']]) -> 'torch.device':
        import torch
        if isinstance(device, torch.device):
            return device
        if device is None or device == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return torch.device("mps")
            return torch.device("cpu")
        if isinstance(device, str):
            return torch.device(device)
        if isinstance(device, int):
            if device >= 0 and torch.cuda.is_available():
                return torch.device(f"cuda:{device}")
            return torch.device("cpu")
        return torch.device("cpu")

class AnonymizationBuilder:
    def __init__(self, anonymizer: 'Anonymizer', model_name: str):
        from dp.methods.registry import get_capabilities
        self.anonymizer = anonymizer
        self.model_name = model_name
        self.capabilities = get_capabilities(model_name)

    def set_dataset_records(self, dataset_records: Iterable[DatasetRecord]):
        self.anonymizer.set_dataset_records(dataset_records)
        return self
    
    def set_explainer(self, explainer: TokenExplainer):
        self.anonymizer.set_explainer(explainer)
        return self
    
    def set_selector(self, selector: AnonymizerUnit):
        self.anonymizer.set_selector(selector)
        return self
    
    def set_output_handler(self, output_handler: 'OutputHandler'):
        self.anonymizer.set_output_handler(output_handler)
        return self
    
    def set_splitter(self, splitter: TextSplitter):
        self.anonymizer.set_splitter(splitter)
        return self
    
    def build(self) -> 'Anonymizer':
        return self.anonymizer