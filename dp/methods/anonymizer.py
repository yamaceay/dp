from typing import Dict, Iterator, List, Optional, Iterable, Union, TYPE_CHECKING, Tuple
from abc import ABC
from dataclasses import dataclass, field
from tqdm import tqdm

from dp.methods.constants import Buckets, BucketDict
from dp.loaders.base import DatasetRecord, TextAnnotation, TextAnnotations
from dp.utils.explainer.base import TokenExplainer
from dp.utils.selector.base import AnonymizerUnit
from dp.utils.splitter import TextSplitter
from dp.utils.device import resolve_device

if TYPE_CHECKING:
    from dp.utils.output import OutputHandler

@dataclass
class AnonymizationResult:
    text: str
    annotations: TextAnnotations = field(default_factory=TextAnnotations)
    metadata: dict = field(default_factory=dict)

class Anonymizer(ABC):
    _splitter: Optional[TextSplitter]

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
        self._splitter: Optional[TextSplitter] = None
        self.device = resolve_device(kwargs.get("device"))

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

    def pre_stream_anonymize(self, texts_or_indices: Optional[Union[List[str], List[int]]] = None, risk_scores: Optional[Dict[str, Dict[str, object]]] = None, *args, **kwargs) -> None:
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

    def _spans_for_indices(
        self,
        text: str,
        offsets: List[Tuple[int, int]],
        indices: List[int],
        replacements: Dict[int, str],
        labels: Dict[int, str],
    ) -> List[TextAnnotation]:
        if not isinstance(indices, list):
            raise ValueError("indices must be a list of ints")
        out: List[TextAnnotation] = []
        for idx in indices:
            if not isinstance(idx, int):
                raise ValueError("indices must be a list of ints")
            if idx < 0 or idx >= len(offsets):
                raise IndexError(f"index {idx} is out of bounds")
            start, end = offsets[idx]
            repl = replacements.get(idx)
            label = labels.get(idx)
            if not isinstance(repl, str) or not repl:
                raise ValueError("Missing replacement for span index")
            if not isinstance(label, str) or not label:
                raise ValueError("Missing label for span index")
            out.append(
                TextAnnotation(
                    start=start,
                    end=end,
                    label=label,
                    text=text[start:end],
                    replacement=repl,
                )
            )
        return out

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