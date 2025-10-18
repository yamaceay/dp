import argparse
import json
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, List

from dp.loaders import get_adapter, DatasetRecord
from dp.utils import TRIDetector


def load_annotations_from_folder(folder_path: str, records: List[DatasetRecord]) -> Dict[str, Dict]:
    """Load annotation files from folder."""
    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"Annotation folder not found: {folder_path}")
    
    method_annotations = {}
    
    for file in os.listdir(folder_path):
        if not file.endswith('.json'):
            continue
        
        method_name = file.rsplit('.', 1)[0]
        file_path = os.path.join(folder_path, file)
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                annotations = json.load(f)
            method_annotations[method_name] = annotations
            print(f"✓ Loaded {method_name} annotations: {len(annotations)} documents")
        except Exception as e:
            print(f"✗ Failed to load {method_name}: {e}")
    
    return method_annotations


def anonymize_text(text: str, annotations: List, mask_token: str = "[MASK]") -> str:
    """Anonymize text using annotations."""
    if not annotations:
        return text
    
    spans = []
    for annot in annotations:
        if isinstance(annot, dict):
            start = annot.get('start', annot.get('start_offset'))
            end = annot.get('end', annot.get('end_offset'))
            if start is not None and end is not None:
                spans.append((int(start), int(end)))
        elif isinstance(annot, (list, tuple)) and len(annot) >= 2:
            spans.append((int(annot[0]), int(annot[1])))
    
    spans.sort(reverse=True)
    
    anonymized = text
    for start, end in spans:
        if 0 <= start < end <= len(anonymized):
            anonymized = anonymized[:start] + mask_token + anonymized[end:]
    
    return anonymized


def create_eval_datasets(records: List[DatasetRecord], 
                        annotation_methods: Dict[str, Dict]) -> Dict[str, List[DatasetRecord]]:
    """Create evaluation datasets for each anonymization method."""
    eval_datasets = {}
    
    for method_name, method_annotations in annotation_methods.items():
        eval_records = []
        for record in records:
            doc_annotations = method_annotations.get(record.uid, [])
            anonymized_text = anonymize_text(record.text, doc_annotations)
            
            eval_record = DatasetRecord(
                uid=record.uid,
                text=anonymized_text,
                name=record.name,
                spans=record.spans,
                utilities=record.utilities,
                metadata=record.metadata
            )
            eval_records.append(eval_record)
        
        eval_datasets[method_name] = eval_records
        print(f"✓ Created {method_name} eval dataset: {len(eval_records)} records")
    
    return eval_datasets


def main():
    parser = argparse.ArgumentParser(description="Train and evaluate TRI model for re-identification")
    parser.add_argument("--dataset", type=str, default="tab", choices=["tab", "trustpilot", "db_bio"], 
                        help="Dataset name")
    parser.add_argument("--data-path", type=str, default="data/TAB/tab.json",
                        help="Path to dataset file")
    parser.add_argument("--annotation-folder", type=str, 
                        default="/Users/yay/work/DPMLM/outputs/tab/samples/train_100/annotations/simple",
                        help="Path to folder with annotation JSON files for evaluation")
    parser.add_argument("--best-metric-dataset", type=str, default=None,
                        help="Which evaluation dataset to use for best model selection (e.g., 'spacy', 'manual', 'presidio')")
    parser.add_argument("--model-name", type=str, default="distilbert-base-uncased",
                        help="Base model name")
    parser.add_argument("--max-records", type=int, default=None,
                        help="Maximum number of records to use")
    parser.add_argument("--finetuning-epochs", type=int, default=15,
                        help="Number of finetuning epochs")
    parser.add_argument("--pretraining-epochs", type=int, default=3,
                        help="Number of pretraining epochs")
    parser.add_argument("--batch-size", type=int, default=16,
                        help="Batch size for training")
    parser.add_argument("--pretraining-batch-size", type=int, default=8,
                        help="Batch size for pretraining")
    parser.add_argument("--use-pretraining", action="store_true",
                        help="Use MLM pretraining before finetuning")
    parser.add_argument("--mode", type=str, default="train", choices=["train", "evaluate", "predict"],
                        help="Mode: train, evaluate, or predict")
    parser.add_argument("--model-path", type=str, default=None,
                        help="Path to load pretrained model (for evaluate/predict modes)")
    
    args = parser.parse_args()
    
    print(f"Loading {args.dataset} dataset from {args.data_path}...")
    
    if args.dataset == "tab":
        adapter = get_adapter(args.dataset, data_in=args.data_path, max_records=args.max_records)
    else:
        adapter = get_adapter(args.dataset, data_path=args.data_path, max_records=args.max_records)
    
    records = list(adapter.iter_records())
    print(f"✓ Loaded {len(records)} records")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = args.model_path or f"models/tri_pipelines/{args.dataset}/{timestamp}"
    
    if args.mode == "train":
        print(f"\nInitializing TRI detector for {args.dataset}...")
        tri = TRIDetector(
            dataset_name=args.dataset,
            model_name=args.model_name,
            max_length=512
        )
        
        tri.set_train_dataset(records)
        
        eval_datasets = {}
        if args.annotation_folder:
            print(f"\nLoading annotations from {args.annotation_folder}...")
            annotation_methods = load_annotations_from_folder(args.annotation_folder, records)
            eval_datasets = create_eval_datasets(records, annotation_methods)
        else:
            print("\nNo annotation folder provided, using original texts for evaluation")
            eval_datasets = {"original": records}
        
        tri.set_eval_datasets(eval_datasets)
        
        print(f"\nFinetuning for {args.finetuning_epochs} epochs...")
        tri.train(
            epochs=args.finetuning_epochs,
            batch_size=args.batch_size,
            output_dir=model_path,
            use_pretraining=args.use_pretraining,
            pretraining_epochs=args.pretraining_epochs,
            best_metric_dataset=args.best_metric_dataset,
        )
        
        print(f"\n✓ Model saved to {model_path}")
        
    elif args.mode == "evaluate":
        print(f"\nLoading model from {model_path}...")
        tri = TRIDetector(
            dataset_name=args.dataset,
            model_name=args.model_name,
            max_length=512
        )
        tri.load(model_path)
        
        if args.annotation_folder:
            print(f"\nLoading annotations from {args.annotation_folder}...")
            annotation_methods = load_annotations_from_folder(args.annotation_folder, records)
            eval_datasets = create_eval_datasets(records, annotation_methods)
            
            for method_name, eval_records in eval_datasets.items():
                print(f"\nEvaluating on {method_name}...")
                results = tri.evaluate(eval_records)
                print(f"✓ {method_name} Results: {results}")
        else:
            print("\nEvaluating on original texts...")
            results = tri.evaluate(records)
            print(f"\n✓ Results: {results}")
        
    elif args.mode == "predict":
        print(f"\nLoading model from {model_path}...")
        tri = TRIDetector(
            dataset_name=args.dataset,
            model_name=args.model_name,
            max_length=512
        )
        tri.load(model_path)
        
        print("\nPredicting on sample records...")
        predictions = tri.predict(records[:5])
        
        print("\n✓ Sample predictions:")
        for uid, probs in list(predictions.items())[:5]:
            top_label = max(probs.items(), key=lambda x: x[1])
            print(f"  Record {uid}: {top_label[0]} ({top_label[1]:.2%})")

if __name__ == "__main__":
    main()
