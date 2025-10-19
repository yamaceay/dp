import argparse
import os
from datetime import datetime
from typing import List

from dp.loaders import get_adapter, DatasetRecord
from dp.utils.pii_detector import PIIDetector


def main():
    parser = argparse.ArgumentParser(description="Train and evaluate PII detection models")
    parser.add_argument("--dataset", type=str, default="tab", choices=["tab", "trustpilot", "db_bio"], 
                        help="Dataset name")
    parser.add_argument("--data-path", type=str, default="data/TAB/splitted",
                        help="Path to dataset file or directory")
    parser.add_argument("--model-name", type=str, default="roberta-base",
                        help="Base model name for PII detection")
    parser.add_argument("--epochs", type=int, default=5,
                        help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=8,
                        help="Batch size for training")
    parser.add_argument("--max-records", type=int, default=None,
                        help="Maximum number of records to use")
    parser.add_argument("--use-nervaluate", action="store_true", default=True,
                        help="Use nervaluate for sophisticated NER evaluation")
    parser.add_argument("--evaluation-mode", type=str, default="partial", 
                        choices=["strict", "partial", "exact"],
                        help="Evaluation mode for nervaluate (strict, partial, exact)")
    parser.add_argument("--metric-mode", type=str, default="recall", 
                        choices=["precision", "recall", "f1"],
                        help="Metric mode for selecting best model during training")
    parser.add_argument("--mode", type=str, default="train", choices=["train", "evaluate", "predict"],
                        help="Mode: train, evaluate, or predict")
    parser.add_argument("--model-path", type=str, default=None,
                        help="Path to load pretrained model (for evaluate/predict modes)")
    
    args = parser.parse_args()
    
    print(f"Loading {args.dataset} dataset from {args.data_path}...")
    
    train_dataset, val_dataset, test_dataset = None, None, None
    
    if os.path.isdir(args.data_path):
        train_path = os.path.join(args.data_path, "train.json")
        val_path = os.path.join(args.data_path, "dev.json")
        test_path = os.path.join(args.data_path, "test.json")
        
        if os.path.exists(train_path):
            adapter = get_adapter(args.dataset, data_in=train_path, max_records=args.max_records)
            train_dataset = list(adapter.iter_records())
            print(f"✓ Loaded {len(train_dataset)} training records")
        
        if os.path.exists(val_path):
            adapter = get_adapter(args.dataset, data_in=val_path, max_records=args.max_records)
            val_dataset = list(adapter.iter_records())
            print(f"✓ Loaded {len(val_dataset)} validation records")
        
        if os.path.exists(test_path):
            adapter = get_adapter(args.dataset, data_in=test_path, max_records=args.max_records)
            test_dataset = list(adapter.iter_records())
            print(f"✓ Loaded {len(test_dataset)} test records")
    else:
        adapter = get_adapter(args.dataset, data_in=args.data_path, max_records=args.max_records)
        all_records = list(adapter.iter_records())
        print(f"✓ Loaded {len(all_records)} records (no train/val/test split)")
        test_dataset = all_records
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = args.model_path or f"models/pii_detectors/{args.dataset}/{timestamp}"
    
    if args.mode == "train":
        if train_dataset is None:
            raise ValueError("Training dataset not found. Provide a directory with train.json")
        
        print(f"\nInitializing PII detector...")
        detector = PIIDetector(model_name=args.model_name)
        
        detector.set_train_dataset(train_dataset)
        if val_dataset:
            detector.set_val_dataset(val_dataset)
        if test_dataset:
            detector.set_test_dataset(test_dataset)
        
        print(f"\nTraining for {args.epochs} epochs...")
        detector.train(
            epochs=args.epochs,
            batch_size=args.batch_size,
            output_dir=model_path,
            use_nervaluate=args.use_nervaluate,
            nervaluate_mode=args.evaluation_mode,
            metric_mode=args.metric_mode,
        )
        
        print(f"\n✓ Model saved to {model_path}")
        
        if test_dataset:
            print("\nEvaluating on test set...")
            metrics = detector.evaluate(
                test_dataset, 
                use_nervaluate=args.use_nervaluate,
                modes=["strict", "partial", "exact"]
            )
            
            print("\nFinal Test Results:")
            for key, value in metrics.items():
                if key != "per_category" and not isinstance(value, dict):
                    print(f"  {key}: {value}")
        
    elif args.mode == "evaluate":
        if test_dataset is None:
            raise ValueError("Test dataset not found for evaluation")
        
        print(f"\nLoading model from {model_path}...")
        detector = PIIDetector(model_name=model_path)  # Load from path
        
        print("\nEvaluating on test set...")
        metrics = detector.evaluate(
            test_dataset,
            use_nervaluate=args.use_nervaluate,
            modes=["strict", "partial", "exact"]
        )

        print("\nEvaluation Results:")
        for key, value in metrics.items():
            if key != "per_category" and not isinstance(value, dict):
                print(f"  {key}: {value}")
        
    elif args.mode == "predict":
        if test_dataset is None:
            raise ValueError("Dataset not found for prediction")
        
        print(f"\nLoading model from {model_path}...")
        detector = PIIDetector(model_name=model_path)  # Load from path

        print("\nPredicting on sample records...")
        sample_records = test_dataset[:5]
        predictions = detector.predict(sample_records)
        
        print("\n✓ Sample predictions:")
        for pred in predictions:
            print(f"\n  Record: {pred.uid}")
            print(f"  Text: {pred.text[:100]}..." if len(pred.text) > 100 else f"  Text: {pred.text}")
            if pred.spans:
                print(f"  Detected spans:")
                for span in pred.spans:
                    print(f"    - [{span.start}:{span.end}] {span.text!r} -> {span.label} (conf: {span.confidence:.2f})")
            else:
                print(f"  No spans detected")


if __name__ == "__main__":
    main()
