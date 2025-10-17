"""
PII Detector for identifying personally identifiable information in text.

This module provides a PIIDetector class that can train, predict, and evaluate
PII detection models using transformer-based token classification.
"""

from typing import List, Optional, Dict, Any
import torch
import numpy as np
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification,
    pipeline,
)
from torch.utils.data import Dataset
from sklearn.metrics import precision_recall_fscore_support, classification_report

from dp.loaders.base import DatasetRecord, TextAnnotation


class PIIDetector:
    """
    PII Detector for training and using token classification models.
    
    This class provides a complete pipeline for PII detection:
    - Training on labeled datasets
    - Prediction on unlabeled text
    - Evaluation against ground truth
    
    Usage:
        # Initialize detector
        detector = PIIDetector(
            model_name="distilbert-base-uncased",
            labels=["O", "B-PERSON", "I-PERSON", "B-EMAIL", "I-EMAIL"]
        )
        
        # Set datasets
        detector.set_train_dataset(train_records)
        detector.set_val_dataset(val_records)
        detector.set_test_dataset(test_records)
        
        # Train
        detector.train(epochs=3, output_dir="./pii_model")
        
        # Predict
        predictions = detector.predict(unlabeled_records)
        
        # Evaluate
        metrics = detector.evaluate(test_records)
    
    Args:
        model_name: HuggingFace model name (default: "distilbert-base-uncased")
        labels: List of BIO labels (default: basic PII labels)
        max_length: Maximum sequence length (default: 512)
        device: Device to use ("auto", "cpu", "cuda", "mps")
    """
    
    def __init__(
        self,
        model_name: str = "distilbert-base-uncased",
        labels: Optional[List[str]] = None,
        max_length: int = 512,
        device: str = "auto",
    ):
        self.model_name = model_name
        self.max_length = max_length
        
        self.device = self.resolve_device(device)
        
        self.labels = []
        self.label_to_id = {}
        self.id_to_label = {}
        self.tokenizer = None
        self.model = None

        if labels is not None:
            self.labels = labels
            self._initialize_model_and_tokenizer(labels)
        
        self.train_records: Optional[List[DatasetRecord]] = None
        self.val_records: Optional[List[DatasetRecord]] = None
        self.test_records: Optional[List[DatasetRecord]] = None

        if labels is not None:
            print(f"✓ PIIDetector initialized with {len(labels)} labels on {self.device}")
        else:
            print(f"✓ PIIDetector initialized without labels on {self.device}")
    
    def set_train_dataset(self, records: List[DatasetRecord]) -> None:
        """Set training dataset."""
        self.train_records = records
        if not self.labels:
            self.labels = self._get_labels(records)
            self._initialize_model_and_tokenizer(self.labels)
        print(f"✓ Training dataset set: {len(records)} records")
    
    def set_val_dataset(self, records: List[DatasetRecord]) -> None:
        """Set validation dataset."""
        self.val_records = records
        if not self.labels:
            self.labels = self._get_labels(records)
            self._initialize_model_and_tokenizer(self.labels)
        print(f"✓ Validation dataset set: {len(records)} records")
    
    def set_test_dataset(self, records: List[DatasetRecord]) -> None:
        """Set test dataset."""
        self.test_records = records
        if not self.labels:
            self.labels = self._get_labels(records)
            self._initialize_model_and_tokenizer(self.labels)
        print(f"✓ Test dataset set: {len(records)} records")
    
    def _get_labels(self, records: List[DatasetRecord]) -> List[str]:
        """Extract unique BIO labels from dataset records."""
        labels = set()
        for record in records:
            spans = record.spans or []
            for span in spans:
                labels.add(span.label)
        if None in labels:
            labels.remove(None)
        labels = list(labels)
        b_labels = [f"B-{label}" for label in labels]
        i_labels = [f"I-{label}" for label in labels]
        return ["O"] + b_labels + i_labels

    def resolve_device(self, device: str) -> torch.device:
        if device == "auto":
            if torch.backends.mps.is_available():
                device = torch.device("mps")
            elif torch.cuda.is_available():
                device = torch.device("cuda")
            else:
                device = torch.device("cpu")
        else:
            device = torch.device(device)
        return device

    def _initialize_model_and_tokenizer(self, labels: List[str]) -> None:
        self.labels = labels
        self.label_to_id = {label: idx for idx, label in enumerate(labels)}
        self.id_to_label = {idx: label for label, idx in self.label_to_id.items()}
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForTokenClassification.from_pretrained(
            self.model_name,
            num_labels=len(labels),
            id2label=self.id_to_label,
            label2id=self.label_to_id,
            ignore_mismatched_sizes=True,
        ).to(self.device)

    def train(
        self,
        epochs: int = 3,
        output_dir: str = "./pii_model_output",
        batch_size: int = 8,
        learning_rate: float = 2e-5,
        **kwargs
    ) -> None:
        """
        Train the PII detection model.
        
        Args:
            epochs: Number of training epochs
            output_dir: Directory to save model checkpoints
            batch_size: Training batch size
            learning_rate: Learning rate
            **kwargs: Additional training arguments
        """
        if self.train_records is None:
            raise ValueError("Training dataset not set. Use set_train_dataset() first.")
        
        train_dataset = PIIDataset(
            self.train_records, self.tokenizer, self.label_to_id, self.max_length
        )
        
        eval_dataset = None
        if self.val_records:
            eval_dataset = PIIDataset(
                self.val_records, self.tokenizer, self.label_to_id, self.max_length
            )
        
        data_collator = DataCollatorForTokenClassification(
            tokenizer=self.tokenizer,
            padding=True,
        )
        
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size * 2,
            learning_rate=learning_rate,
            warmup_steps=100,
            weight_decay=0.01,
            logging_dir=f"{output_dir}/logs",
            logging_steps=50,
            eval_strategy="steps" if eval_dataset else "no",
            eval_steps=250 if eval_dataset else None,
            save_strategy="steps",
            save_steps=250,
            load_best_model_at_end=True if eval_dataset else False,
            metric_for_best_model="f1" if eval_dataset else None,
            report_to="none",
            fp16=torch.cuda.is_available(),
            **kwargs,
        )
        
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
            compute_metrics=self._compute_metrics,
        )
        
        print(f"Starting training for {epochs} epochs...")
        trainer.train()
        
        trainer.save_model(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        print(f"Training complete! Model saved to {output_dir}")
    
    def predict(self, records: List[DatasetRecord]) -> List[DatasetRecord]:
        """
        Predict PII spans for records with empty spans.
        
        Args:
            records: List of DatasetRecord with empty spans
        
        Returns:
            List of DatasetRecord with predicted spans filled in
        """
        if not records:
            return []
        
        if self.labels is None:
            self.labels = self._get_labels(records)
            self._initialize_model_and_tokenizer(self.labels)
        
        pipe = pipeline(
            "token-classification",
            model=self.model,
            tokenizer=self.tokenizer,
            device=self.device,
            aggregation_strategy="simple",
        )
        
        results = []
        
        for record in records:
            predictions = pipe(record.text)
            
            spans = []
            for pred in predictions:
                spans.append(
                    TextAnnotation(
                        start=pred["start"],
                        end=pred["end"],
                        label=pred["entity_group"],
                        text=record.text[pred["start"]:pred["end"]],
                        confidence=pred["score"],
                    )
                )
            
            new_record = DatasetRecord(
                uid=record.uid,
                text=record.text,
                name=record.name,
                spans=spans,
                utilities=record.utilities.copy(),
                metadata={**record.metadata, "predicted": True},
            )
            results.append(new_record)
        
        print(f"✓ Predicted spans for {len(results)} records")
        return results
    
    def evaluate(self, records: List[DatasetRecord]) -> Dict[str, Any]:
        """
        Evaluate model performance against ground truth.
        
        Args:
            records: List of DatasetRecord with ground truth spans
        
        Returns:
            Dictionary containing evaluation metrics
        """
        if not records:
            raise ValueError("No records provided for evaluation")

        if self.labels is None:
            self.labels = self._get_labels(records)
            self._initialize_model_and_tokenizer(self.labels)

        predicted_records = self.predict(records)

        all_true_labels = []
        all_pred_labels = []
        
        for true_record, pred_record in zip(records, predicted_records):
            true_labels = self._spans_to_char_labels(
                true_record.text, true_record.spans or []
            )
            pred_labels = self._spans_to_char_labels(
                pred_record.text, pred_record.spans or []
            )
            
            all_true_labels.extend(true_labels)
            all_pred_labels.extend(pred_labels)
        
        unique_labels = [l for l in self.labels if l != "O"]
        
        precision, recall, f1, support = precision_recall_fscore_support(
            all_true_labels,
            all_pred_labels,
            labels=unique_labels,
            average="micro",
            zero_division=0,
        )
        
        report = classification_report(
            all_true_labels,
            all_pred_labels,
            labels=unique_labels,
            zero_division=0,
            output_dict=True,
        )
        
        metrics = {
            "precision": float(precision) if precision is not None else 0.0,
            "recall": float(recall) if recall is not None else 0.0,
            "f1": float(f1) if f1 is not None else 0.0,
            "support": int(support) if support is not None else 0,
            "detailed_report": report,
        }
        
        print(f"📊 Evaluation Results:")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall: {recall:.4f}")
        print(f"  F1 Score: {f1:.4f}")
        print(f"  Support: {support}")
        
        return metrics
    
    def _compute_metrics(self, eval_pred):
        """Compute metrics during training."""
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=2)
        
        true_predictions = []
        true_labels = []
        
        for pred_seq, label_seq in zip(predictions, labels):
            for pred, label in zip(pred_seq, label_seq):
                if label != -100:
                    true_predictions.append(self.id_to_label[pred])
                    true_labels.append(self.id_to_label[label])
        
        unique_labels = [l for l in self.labels if l != "O"]
        
        precision, recall, f1, _ = precision_recall_fscore_support(
            true_labels,
            true_predictions,
            labels=unique_labels,
            average="micro",
            zero_division=0,
        )
        
        return {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
        }
    
    def _spans_to_char_labels(
        self, text: str, spans: List[TextAnnotation]
    ) -> List[str]:
        """Convert span annotations to character-level BIO labels."""
        labels = ["O"] * len(text)
        
        for span in sorted(spans, key=lambda s: s.start):
            if span.start >= len(text):
                continue
            
            label = span.label or "ENTITY"
            
            if label.startswith("B-") or label.startswith("I-"):
                label = label[2:]
            
            labels[span.start] = f"B-{label}"
            
            for i in range(span.start + 1, min(span.end, len(text))):
                labels[i] = f"I-{label}"
        
        return labels


class PIIDataset(Dataset):
    """PyTorch Dataset for PII detection."""
    
    def __init__(
        self,
        records: List[DatasetRecord],
        tokenizer,
        label_to_id: Dict[str, int],
        max_length: int = 512,
    ):
        self.records = records
        self.tokenizer = tokenizer
        self.label_to_id = label_to_id
        self.max_length = max_length
    
    def __len__(self):
        return len(self.records)
    
    def __getitem__(self, idx):
        record = self.records[idx]
        text = record.text
        spans = record.spans or []
        
        tokenized = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_offsets_mapping=True,
        )
        
        char_labels = self._spans_to_char_labels(text, spans)
        
        token_labels = []
        for start_char, end_char in tokenized["offset_mapping"]:
            if start_char == 0 and end_char == 0:
                token_labels.append(-100)
            else:
                if start_char < len(char_labels):
                    label = char_labels[start_char]
                    token_labels.append(self.label_to_id.get(label, self.label_to_id["O"]))
                else:
                    token_labels.append(self.label_to_id["O"])
        
        return {
            "input_ids": torch.tensor(tokenized["input_ids"]),
            "attention_mask": torch.tensor(tokenized["attention_mask"]),
            "labels": torch.tensor(token_labels),
        }
    
    def _spans_to_char_labels(
        self, text: str, spans: List[TextAnnotation]
    ) -> List[str]:
        """Convert span annotations to character-level BIO labels."""
        labels = ["O"] * len(text)
        
        for span in sorted(spans, key=lambda s: s.start):
            if span.start >= len(text):
                continue
            
            label = span.label or "ENTITY"
            
            if label.startswith("B-") or label.startswith("I-"):
                label = label[2:]
            
            labels[span.start] = f"B-{label}"
            
            for i in range(span.start + 1, min(span.end, len(text))):
                labels[i] = f"I-{label}"
        
        return labels