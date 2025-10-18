from typing import List, Optional, Dict, Any
import os
import json
import torch
import pandas as pd
from pathlib import Path
from torch.utils.data import Dataset
from torch.nn import functional as F
from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoModelForSequenceClassification,
    AutoModelForMaskedLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    get_constant_schedule,
    pipeline,
)
from torch.optim import AdamW
from collections import OrderedDict
from dp.loaders.base import DatasetRecord


def compute_metrics(eval_pred):
    """Compute accuracy by aggregating logits per individual."""
    logits, labels = eval_pred
    logits = torch.from_numpy(logits)
    
    logits_dict = {}
    for logit, label in zip(logits, labels):
        current_logits = logits_dict.get(label, torch.zeros_like(logit))
        logits_dict[label] = current_logits + logit
    
    num_predictions = len(logits_dict)
    all_predictions = torch.zeros(num_predictions, device="cpu")
    all_labels = torch.zeros(num_predictions, device="cpu")
    
    for idx, (label, summed_logits) in enumerate(logits_dict.items()):
        all_labels[idx] = label
        probabilities = F.softmax(summed_logits, dim=-1)
        all_predictions[idx] = torch.argmax(probabilities)
    
    correct_predictions = torch.sum(all_predictions == all_labels)
    accuracy = (float(correct_predictions) / num_predictions) * 100
    
    return {"Accuracy": accuracy}

class TRIDetector:
    def __init__(
        self,
        dataset_name: str = None,
        model_name: str = "distilbert-base-uncased",
        max_length: int = 512,
        device: str = "auto",
        use_chunking: bool = False,
    ):
        self.dataset_name = dataset_name
        self.model_name = model_name
        self.max_length = max_length
        self.device = self.resolve_device(device)
        self.use_chunking = use_chunking
        self.chunker = None
        
        self.tokenizer = None
        self.model = None
        self.label_to_name = {}
        self.name_to_label = {}
        self.num_labels = 0
        
        self.train_records: Optional[List[DatasetRecord]] = None
        self.eval_records_dict: Optional[Dict[str, List[DatasetRecord]]] = None
        
        if dataset_name is not None:
            print(f"✓ TRIDetector initialized for dataset '{dataset_name}' on {self.device}")
    
    def resolve_device(self, device: str) -> torch.device:
        if device == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return torch.device("mps")
            return torch.device("cpu")
        return torch.device(device)
    
    def set_train_dataset(self, records: List[DatasetRecord]) -> None:
        self.train_records = records
        self._build_label_mappings()
        print(f"✓ Training dataset set: {len(records)} records, {self.num_labels} individuals")
    
    def set_eval_datasets(self, records_dict: Dict[str, List[DatasetRecord]]) -> None:
        self.eval_records_dict = records_dict
        print(f"✓ Evaluation datasets set: {list(records_dict.keys())}")
    
    def _build_label_mappings(self):
        if not self.train_records:
            return
        
        names = sorted(set(r.name for r in self.train_records))
        self.label_to_name = {idx: name for idx, name in enumerate(names)}
        self.name_to_label = {name: idx for idx, name in self.label_to_name.items()}
        self.num_labels = len(names)
    
    def _initialize_tokenizer_and_model(self):
        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if self.model is None and self.num_labels > 0:
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name, num_labels=self.num_labels
            )
            self.model.to(self.device)
    
    def train(
        self,
        epochs: int = 3,
        batch_size: int = 16,
        learning_rate: float = 5e-5,
        output_dir: Optional[str] = None,
        use_pretraining: bool = False,
        pretraining_epochs: int = 3,
        best_metric_dataset: Optional[str] = None,
        **kwargs
    ) -> None:
        if not self.train_records:
            raise ValueError("No training data set. Call set_train_dataset() first.")
        
        if output_dir is None:
            output_dir = f"models/tri_pipelines/{self.dataset_name}"
        
        os.makedirs(output_dir, exist_ok=True)
        
        self._initialize_tokenizer_and_model()
        
        if use_pretraining:
            print(f"Starting additional pretraining for {pretraining_epochs} epochs...")
            self._pretrain(pretraining_epochs, batch_size, learning_rate, output_dir)
        
        print(f"Starting finetuning for {epochs} epochs...")
        train_dataset = TRIDataset(self.train_records, self.tokenizer, self.name_to_label, self.max_length)
        
        eval_datasets_dict = None
        if self.eval_records_dict:
            eval_datasets_dict = {
                name: TRIDataset(records, self.tokenizer, self.name_to_label, self.max_length)
                for name, records in self.eval_records_dict.items()
            }
        
        if best_metric_dataset:
            if eval_datasets_dict and best_metric_dataset in eval_datasets_dict:
                metric_for_best = f"eval_{best_metric_dataset}_Accuracy"
                print(f"📊 Using {best_metric_dataset} dataset for best model selection")
            else:
                available = list(eval_datasets_dict.keys()) if eval_datasets_dict else []
                raise ValueError(f"Best metric dataset '{best_metric_dataset}' not found in eval datasets: {available}")
        else:
            first_eval_name = list(eval_datasets_dict.keys())[0] if eval_datasets_dict else None
            metric_for_best = f"eval_{first_eval_name}_Accuracy" if first_eval_name else None
            if first_eval_name:
                print(f"📊 Using {first_eval_name} dataset for best model selection (default)")
        
        training_args = TrainingArguments(
            output_dir=f"{output_dir}/finetuning",
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            learning_rate=learning_rate,
            logging_strategy="epoch",
            eval_strategy="epoch" if eval_datasets_dict else "no",
            save_strategy="epoch",
            load_best_model_at_end=True if eval_datasets_dict else False,
            metric_for_best_model=metric_for_best,
            greater_is_better=True,
            save_total_limit=1,
            report_to="none",
        )
        
        optimizer = AdamW(self.model.parameters(), lr=learning_rate)
        scheduler = get_constant_schedule(optimizer)
        
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_datasets_dict,
            optimizers=[optimizer, scheduler],
            compute_metrics=compute_metrics,
        )
        
        trainer.train()
        
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        
        os.makedirs(f"{output_dir}/finetuning", exist_ok=True)
        
        label_mapping_path = Path(output_dir) / "label_mapping.json"
        with open(label_mapping_path, "w") as f:
            json.dump(self.name_to_label, f, indent=2)
        
        print(f"✓ Training complete! Model saved to {output_dir}")
    
    def _pretrain(self, epochs: int, batch_size: int, learning_rate: float, output_dir: str):
        """Perform MLM pretraining and update the base model."""
        print(f"📚 Creating MLM model for pretraining...")
        mlm_model = AutoModelForMaskedLM.from_pretrained(self.model_name)
        
        if hasattr(self.model, 'distilbert'):
            mlm_model.distilbert = self.model.distilbert
        elif hasattr(self.model, 'bert'):
            mlm_model.bert = self.model.bert
        elif hasattr(self.model, 'roberta'):
            mlm_model.roberta = self.model.roberta
        
        mlm_model.to(self.device)
        
        train_dataset = TRIDataset(self.train_records, self.tokenizer, self.name_to_label, self.max_length, use_labels=False)
        data_collator = DataCollatorForLanguageModeling(self.tokenizer, mlm_probability=0.15)
        
        training_args = TrainingArguments(
            output_dir=f"{output_dir}/pretraining",
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            learning_rate=learning_rate,
            logging_strategy="epoch",
            save_strategy="no",
            report_to="none",
        )
        
        trainer = Trainer(
            model=mlm_model,
            args=training_args,
            train_dataset=train_dataset,
            data_collator=data_collator,
        )
        
        print(f"🔄 Running MLM pretraining for {epochs} epochs...")
        trainer.train()
        
        if hasattr(self.model, 'distilbert'):
            self.model.distilbert = mlm_model.distilbert
        elif hasattr(self.model, 'bert'):
            self.model.bert = mlm_model.bert
        elif hasattr(self.model, 'roberta'):
            self.model.roberta = mlm_model.roberta
        
        print(f"✓ Pretraining complete, weights transferred to classification model")
        
        if hasattr(mlm_model, 'distilbert'):
            self.model.distilbert.load_state_dict(mlm_model.distilbert.state_dict())
        elif hasattr(mlm_model, 'roberta'):
            base_dict = mlm_model.roberta.state_dict()
            base_dict = {k: v for k, v in base_dict.items() if not k.startswith('pooler')}
            self.model.roberta.load_state_dict(base_dict, strict=False)
        elif hasattr(mlm_model, 'bert'):
            self.model.bert.load_state_dict(mlm_model.bert.state_dict())
        
        del mlm_model
        torch.cuda.empty_cache()
        print("✓ Pretraining complete")
    
    def predict(self, records: List[DatasetRecord]) -> Dict[str, float]:
        if self.model is None:
            raise ValueError("Model not initialized. Train or load a model first.")
        
        pipe = pipeline("text-classification", model=self.model, tokenizer=self.tokenizer, device=self.device if self.device.type != "cpu" else -1, top_k=None, truncation=True, max_length=self.max_length)
        
        results = {}
        for record in records:
            if self.use_chunking and self.chunker is not None:
                from dp.utils.chunking import ProbabilityAggregator, process_with_chunking
                aggregator = ProbabilityAggregator()
                
                def classify(text):
                    preds = pipe(text)[0]
                    return {pred['label']: pred['score'] for pred in preds}
                
                pred_dict = process_with_chunking(record.text, self.chunker, classify, aggregator)
            else:
                predictions = pipe(record.text)[0]
                pred_dict = {pred['label']: pred['score'] for pred in predictions}
            
            results[record.uid] = pred_dict
        
        return results
    
    def evaluate(self, records: List[DatasetRecord]) -> Dict[str, Any]:
        if not records:
            raise ValueError("No records provided for evaluation")
        
        pipe = pipeline("text-classification", model=self.model, tokenizer=self.tokenizer, device=self.device if self.device.type != "cpu" else -1, truncation=True, max_length=self.max_length)
        
        correct = 0
        total = 0
        
        for record in records:
            predictions = pipe(record.text)
            predicted_label = predictions[0]['label']
            true_label = f"LABEL_{self.name_to_label[record.name]}"
            
            if predicted_label == true_label:
                correct += 1
            total += 1
        
        accuracy = correct / total if total > 0 else 0
        
        print(f"✓ Evaluation complete: {correct}/{total} correct ({accuracy:.2%})")
        
        return {
            "accuracy": accuracy,
            "correct": correct,
            "total": total,
        }
    
    def load(self, model_path: Optional[str] = None):
        if model_path is None:
            model_path = f"models/tri_pipelines/{self.dataset_name}"
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.model.to(self.device)
        
        self.num_labels = self.model.config.num_labels
        
        label_mapping_path = Path(model_path) / "label_mapping.json"
        if label_mapping_path.exists():
            with open(label_mapping_path, "r") as f:
                self.name_to_label = json.load(f)
        else:
            print(f"Warning: No label mapping found at {label_mapping_path}")
        
        print(f"✓ Model loaded from {model_path}")


class TRIDataset(Dataset):
    def __init__(self, records: List[DatasetRecord], tokenizer, name_to_label: Dict[str, int], max_length: int, use_labels: bool = True):
        self.records = records
        self.tokenizer = tokenizer
        self.name_to_label = name_to_label
        self.max_length = max_length
        self.use_labels = use_labels
        
        self.encodings = tokenizer(
            [r.text for r in records],
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt"
        )
        
        if use_labels:
            self.labels = torch.tensor([name_to_label[r.name] for r in records])
    
    def __len__(self):
        return len(self.records)
    
    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        if self.use_labels:
            item["labels"] = self.labels[idx]
        return item
