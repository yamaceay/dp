from typing import Dict, List, Optional, Any, Union, Tuple
from hashlib import sha256

from dp.methods.anonymizer import AnonymizationResult, Anonymizer
from dp.methods.constants import Buckets, LambdaParams, buckets_to_dicts, BucketDict
from dp.loaders.base import TextAnnotation, DatasetRecord
from dp.utils.token_ledger import TokenLedger

class BaroudAnonymizer(Anonymizer):
    MODEL_NAME = "baroud"

    def __init__(self, *args, pii_annotator: Optional[str] = None, **kwargs):
        super().__init__(*args, model=self.MODEL_NAME, **kwargs)
        
        self.pii_annotator = pii_annotator
        self.pii_detector = None
        self._annotations_cache: Dict[str, List[TextAnnotation]] = {}
        
        if self.pii_annotator:
            from dp.utils.pii_detector import PIIDetector
            self.pii_detector = PIIDetector(model_name=self.pii_annotator, use_chunking=False)
    
    def hash_text(self, text: str) -> str:
        return sha256(text.encode('utf-8')).hexdigest()

    def pre_stream_anonymize(self, texts_or_indices: Union[List[str], List[int]], *args, **kwargs) -> None:
        if not all(isinstance(i, str) for i in texts_or_indices):
            raise ValueError("BaroudAnonymizer requires texts for pre_stream_anonymize.")
        all_annotations = self._predict_pii_annotations(texts_or_indices)
        self._annotations_cache = {self.hash_text(text): anns for text, anns in zip(texts_or_indices, all_annotations)}

    def _predict_pii_annotations(self, texts: List[str]) -> List[List[TextAnnotation]]:
        if not self.pii_detector:
            raise ValueError("PII detector is not configured for BaroudAnonymizer.")
        records = [DatasetRecord(text=text) for text in texts]
        predictions = self.pii_detector.predict(records)
        return [list(pred.spans or []) for pred in predictions]

    def anonymize_any_text(self, text: str, *args, buckets: Buckets = [], **kwargs) -> List[Tuple[BucketDict, AnonymizationResult]]:
        anns = self._annotations_cache[self.hash_text(text)]
        if len(buckets) != 1 or not isinstance(buckets[0], LambdaParams):
            raise ValueError("BaroudAnonymizer only supports LambdaParams for grid anonymization.")
        lambda_params: LambdaParams = buckets[0]
        
        sorted_thresholds = sorted(lambda_params.values(), reverse=True)
        spans = [(ann.start, ann.end) for ann in anns]
        ledger = TokenLedger(text, spans)
        combos = buckets_to_dicts(buckets)

        aggregated: List[Tuple[BucketDict, AnonymizationResult]] = []
        all_result_spans: List[TextAnnotation] = []
        prev_threshold = 1.0

        for threshold, hp in zip(sorted_thresholds, combos):
            for idx, ann in enumerate(anns):
                if ann.confidence is None:
                    continue
                if prev_threshold > ann.confidence >= threshold:
                    ledger.replace(idx, f"[{ann.label}]")
                    all_result_spans.append(
                        TextAnnotation(
                            start=ann.start,
                            end=ann.end,
                            label=ann.label,
                            text=ann.text,
                            replacement=f"[{ann.label}]",
                            confidence=ann.confidence,
                            annotator="baroud",
                        )
                    )
            
            if all_result_spans:
                anonymized_text = ledger.render_offsets(text)
                metadata = {
                    "method": "baroud",
                    "pii_detected": len(all_result_spans),
                    "lambda": threshold,
                }
                aggregated.append((hp, AnonymizationResult(text=anonymized_text, spans=all_result_spans[:], metadata=metadata)))
            
            prev_threshold = threshold
        
        return aggregated