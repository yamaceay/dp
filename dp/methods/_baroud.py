from typing import Any, Dict, List, Optional, Union, Tuple
from hashlib import sha256

from dp.methods.anonymizer import AnonymizationResult, Anonymizer
from dp.methods.constants import Buckets, LambdaParams, buckets_to_dicts, BucketDict
from dp.loaders.base import TextAnnotation, TextAnnotations, TokenEdit, DatasetRecord
from dp.utils.token_ledger import TokenLedger
from dp.utils.selector.base import AnonymizerUnit, ApplyFn, AnonymizationStep
from dp.utils.selector.pii_only_selector import PIIOnlyUnit


class BaroudAnonymizer(Anonymizer):
    MODEL_NAME = "baroud"

    def __init__(self, *args, pii_annotator: Optional[str] = None, mask_text: str = "[{label}]", **kwargs):
        super().__init__(*args, model=self.MODEL_NAME, **kwargs)
        
        self.pii_annotator = pii_annotator
        self.mask_text = mask_text
        self._unit: Optional[PIIOnlyUnit] = None
        self._annotations_cache: Dict[str, List[TextAnnotation]] = {}
        
        if self.pii_annotator:
            from dp.utils.pii_detector import PIIDetector
            pii_detector = PIIDetector(model_name=self.pii_annotator, use_chunking=False)
            self._unit = PIIOnlyUnit(pii_detector=pii_detector)

    def set_unit(self, unit: AnonymizerUnit) -> None:
        self._unit = unit

    def set_filtering_strategy(self, detector: AnonymizerUnit) -> None:
        self._unit = detector
    
    def hash_text(self, text: str) -> str:
        return sha256(text.encode('utf-8')).hexdigest()

    def pre_stream_anonymize(self, texts_or_indices: Union[List[str], List[int]], *args, **kwargs) -> None:
        if not all(isinstance(i, str) for i in texts_or_indices):
            raise ValueError("BaroudAnonymizer requires texts for pre_stream_anonymize.")
        all_annotations = self._predict_pii_annotations(texts_or_indices)
        self._annotations_cache = {self.hash_text(text): anns for text, anns in zip(texts_or_indices, all_annotations)}

    def _predict_pii_annotations(self, texts: List[str]) -> List[List[TextAnnotation]]:
        if self._unit is None or not hasattr(self._unit, 'pii_detector'):
            raise ValueError("PII detector is not configured for BaroudAnonymizer.")
        records = [DatasetRecord(text=text) for text in texts]
        predictions = self._unit.pii_detector.predict(records)
        return [list(pred.spans or []) for pred in predictions]

    def _make_apply_fn(
        self,
        text: str,
        anns: List[TextAnnotation],
        runtime_stats: Dict[str, int],
    ) -> ApplyFn:
        ann_by_span: Dict[Tuple[int, int], TextAnnotation] = {}
        for ann in anns:
            ann_by_span[(ann.start, ann.end)] = ann

        def apply_fn(idx: int, ledger: TokenLedger) -> None:
            entry = ledger.entry(idx)
            span_key = (entry.start, entry.end)
            ann = ann_by_span.get(span_key)
            if ann is None:
                return
            
            label = ann.label or "MASK"
            replacement = self.mask_text.format(label=label)
            ledger.replace(idx, replacement)
            runtime_stats["masked"] += 1

        return apply_fn

    def anonymize_any_text(self, text: str, *args, buckets: Buckets = [], **kwargs) -> List[Tuple[BucketDict, AnonymizationResult]]:
        anns = self._annotations_cache.get(self.hash_text(text), [])
        
        if len(buckets) != 1 or not isinstance(buckets[0], LambdaParams):
            raise ValueError("BaroudAnonymizer only supports LambdaParams for grid anonymization.")
        
        lambda_params: LambdaParams = buckets[0]
        
        if self._unit is None:
            from dp.utils.selector.all_selector import AllUnit
            self._unit = AllUnit()
        
        self._unit.set_thresholds(lambda_params.values(), name="lambda")
        
        spans = [(ann.start, ann.end) for ann in anns]
        
        runtime_stats: Dict[str, int] = {"masked": 0}
        apply_fn = self._make_apply_fn(text, anns, runtime_stats)
        
        outputs: List[Tuple[BucketDict, AnonymizationResult]] = []
        
        for step in self._unit.anonymize(text, spans, apply_fn):
            threshold = step.threshold
            hp: BucketDict = {"lambda": threshold}
            
            private_text = step.text
            ledger = step.ledger
            
            result_spans = [
                TextAnnotation(
                    start=spans[idx][0],
                    end=spans[idx][1],
                    label=anns[idx].label if idx < len(anns) else "MASK",
                    text=text[spans[idx][0]:spans[idx][1]],
                    replacement=self.mask_text.format(label=anns[idx].label if idx < len(anns) else "MASK"),
                    confidence=anns[idx].confidence if idx < len(anns) else None,
                    annotator="baroud",
                )
                for idx in step.new_indices
            ]
            
            metadata: Dict[str, Any] = {
                "method": "baroud",
                "lambda": threshold,
                "pii_detected": runtime_stats["masked"],
                **step.metadata,
            }
            token_edits = [TokenEdit.from_mapping(e) for e in ledger.edits_metadata()]

            annotations = TextAnnotations(token_edits=token_edits)
            annotations.spans = result_spans
            
            outputs.append((
                hp,
                AnonymizationResult(
                    text=private_text,
                    annotations=annotations,
                    metadata=metadata,
                ),
            ))
        
        if not outputs:
            outputs.append((
                {"lambda": 1.0},
                AnonymizationResult(
                    text=text,
                    annotations=TextAnnotations(),
                    metadata={"method": "baroud", "masked": 0},
                ),
            ))
        
        return outputs