# from typing import List, Optional, Dict, Any, Tuple, Union
# import torch
# from transformers import (
#     AutoModelForMaskedLM,
#     TrainingArguments,
#     Trainer,
#     DataCollatorForLanguageModeling,
# )
# from dp.loaders.base import DatasetRecord
# from dp.tri.base import TRIDetector, TRIDataset


# class TRIDetectorWithDeid(TRIDetector):
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.eval_records_dict: Optional[Dict[str, List[DatasetRecord]]] = None
    
#     def setup(self, train_records: List[DatasetRecord], eval_records_dict: Optional[Dict[str, List[DatasetRecord]]] = None) -> None:
#         self._set_train_dataset(train_records)
#         if eval_records_dict:
#             self._set_eval_datasets(eval_records_dict)

#     def _set_train_dataset(self, records: List[DatasetRecord]) -> None:
#         if not records:
#             raise ValueError("Training records cannot be empty")
#         self.train_records = records
#         self.build_label_mappings()

#     def _set_eval_datasets(self, records_dict: Dict[str, List[DatasetRecord]]) -> None:
#         if not records_dict:
#             raise ValueError("Evaluation records dict cannot be empty")
#         for name, records in records_dict.items():
#             if not records:
#                 raise ValueError(f"Evaluation dataset '{name}' has no records")
#         self.eval_records_dict = records_dict

#     def pretrain(self, epochs: int, batch_size: int, learning_rate: float, output_dir: str) -> None:
#         if not self.model:
#             raise ValueError("Model must be initialized before pretraining")
#         if not self.tokenizer:
#             raise ValueError("Tokenizer must be initialized before pretraining")
        
#         mlm_model = AutoModelForMaskedLM.from_pretrained(self.model_name)        
#         if hasattr(self.model, 'distilbert'):
#             mlm_model.distilbert = self.model.distilbert
#         elif hasattr(self.model, 'bert'):
#             mlm_model.bert = self.model.bert
#         elif hasattr(self.model, 'roberta'):
#             mlm_model.roberta = self.model.roberta
#         else:
#             raise ValueError(f"Unsupported model architecture: {self.model_name}")
        
#         mlm_model.to(self.device)
        
#         train_dataset = TRIDataset(self.train_records, self.tokenizer, self.name_to_label, self.max_length, use_labels=False)
#         data_collator = DataCollatorForLanguageModeling(self.tokenizer, mlm_probability=0.15)
        
#         training_args = TrainingArguments(
#             output_dir=f"{output_dir}/pretraining",
#             num_train_epochs=epochs,
#             per_device_train_batch_size=batch_size,
#             learning_rate=learning_rate,
#             logging_strategy="epoch",
#             save_strategy="no",
#             report_to="none",
#         )
        
#         trainer = Trainer(
#             model=mlm_model,
#             args=training_args,
#             train_dataset=train_dataset,
#             data_collator=data_collator,
#         )
        
#         trainer.train()
        
#         if hasattr(mlm_model, 'distilbert'):
#             self.model.distilbert.load_state_dict(mlm_model.distilbert.state_dict())
#         elif hasattr(mlm_model, 'roberta'):
#             base_dict = mlm_model.roberta.state_dict()
#             base_dict = {k: v for k, v in base_dict.items() if not k.startswith('pooler')}
#             self.model.roberta.load_state_dict(base_dict, strict=False)
#         elif hasattr(mlm_model, 'bert'):
#             self.model.bert.load_state_dict(mlm_model.bert.state_dict())
#         else:
#             raise ValueError(f"Unable to transfer weights for {self.model_name}")
        
#         del mlm_model
#         torch.cuda.empty_cache()

#     def get_eval_dataset(self, best_metric_dataset: Optional[str] = None, per_step: Optional[int] = None) -> Tuple[Union[TRIDataset, Dict[str, TRIDataset]], dict[str, Any]]:
#         eval_datasets_dict = None
#         eval_kwargs = {}
#         if not self.eval_records_dict:
#             return None, eval_kwargs

#         eval_datasets_dict = {
#             name: TRIDataset(records, self.tokenizer, self.name_to_label, self.max_length)
#             for name, records in self.eval_records_dict.items()
#         }
    
#         if best_metric_dataset:
#             if best_metric_dataset not in eval_datasets_dict:
#                 available = list(eval_datasets_dict.keys())
#                 raise ValueError(f"best_metric_dataset '{best_metric_dataset}' not found in {available}")
#             metric_for_best = f"eval_{best_metric_dataset}_Accuracy"
#         else:
#             first_eval_name = list(eval_datasets_dict.keys())[0]
#             metric_for_best = f"eval_{first_eval_name}_Accuracy"
        
#         eval_kwargs = {
#             "load_best_model_at_end": True,
#             "metric_for_best_model": metric_for_best,
#             "greater_is_better": True,
#         }
#         if per_step:
#             eval_kwargs.update({
#                 "eval_strategy": "steps",
#                 "eval_steps": per_step,
#             })
#         else:
#             eval_kwargs.update({
#                 "eval_strategy": "epoch",
#             })

#         return eval_datasets_dict, eval_kwargs
