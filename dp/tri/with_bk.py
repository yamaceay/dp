from typing import List, Optional, Dict, Any, Tuple, Union
from dp.loaders.base import DatasetRecord
from dp.tri.base import TRIDetector, TRIDataset
from dp.tri.loaders.base import AttackerDatasetRecord


class TRIDetectorWithBK(TRIDetector):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.eval_records: Optional[List[DatasetRecord]] = None
        self.original_eval_records: Optional[List[DatasetRecord]] = None
    
    def setup(self, records: List[AttackerDatasetRecord], include_original_eval: bool = False) -> None:
        self._set_dataset(records, include_original_eval=include_original_eval)
        self.build_label_mappings()

    def _set_dataset(self, records: List[AttackerDatasetRecord], include_original_eval: bool) -> None:
        if not records:
            raise ValueError("Training records cannot be empty")
        self.train_records, self.eval_records = [], []
        self.original_eval_records = [] if include_original_eval else None
        for record in records:
            eval_record = DatasetRecord(
                uid=record.uid,
                text=record.rewrited_text,
                name=record.name,
                spans=record.spans,
                metadata=record.metadata,
            )
            self.eval_records.append(eval_record)
            if self.original_eval_records is not None:
                self.original_eval_records.append(
                    DatasetRecord(
                        uid=record.uid,
                        text=record.text,
                        name=record.name,
                        spans=record.spans,
                        metadata=record.metadata,
                    )
                )
            for bk_key, bk_value in record.background_knowledge:
                new_metadata = {
                    **record.metadata,
                    "background_knowledge": bk_key
                }
                train_record = DatasetRecord(
                    uid=record.uid,
                    text=bk_value,
                    name=record.name,
                    spans=record.spans,
                    metadata=new_metadata,
                )
                self.train_records.append(train_record)

    def get_eval_dataset(self, best_metric_dataset: Optional[str] = None, per_step: Optional[int] = None) -> Tuple[Union[TRIDataset, Dict[str, TRIDataset]], Dict[str, Any]]:
        if self.eval_records is None:
            raise ValueError("eval_records is not set; call setup() first")
        eval_dataset: Union[TRIDataset, Dict[str, TRIDataset]]
        if self.original_eval_records is not None:
            eval_dataset = {
                "deidentified": TRIDataset(self.eval_records, self.tokenizer, self.name_to_label, self.max_length),
                "original": TRIDataset(self.original_eval_records, self.tokenizer, self.name_to_label, self.max_length),
            }
        else:
            eval_dataset = TRIDataset(self.eval_records, self.tokenizer, self.name_to_label, self.max_length)
            
        eval_kwargs = {
            "load_best_model_at_end": True,
            "metric_for_best_model": "eval_Accuracy",
            "greater_is_better": True,
        }
        if per_step:
            eval_kwargs.update({
                "eval_strategy": "steps",
                "eval_steps": per_step,
            })
        else:
            eval_kwargs.update({
                "eval_strategy": "epoch",
            })

        return eval_dataset, eval_kwargs
