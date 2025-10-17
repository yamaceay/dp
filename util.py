from typing import Optional, List, Tuple
import argparse
import yaml
import os
from datetime import datetime

from dp.loaders import ADAPTER_REGISTRY, DatasetRecord
from dp.utils.pii_detector import PIIDetector

available_datasets = list(ADAPTER_REGISTRY.keys())

def add_data_args(parser: argparse.ArgumentParser) -> Tuple[argparse.ArgumentParser, List[str]]:
    parser.add_argument('--data', type=str, required=True, choices=available_datasets, help='Dataset name (trustpilot, tab, db_bio)')
    parser.add_argument('--data_in', type=str, required=True, help='Path to input data file or directory')
    return parser, ['data', 'data_in']

def add_model_args(parser: argparse.ArgumentParser) -> Tuple[argparse.ArgumentParser, List[str]]:
    parser.add_argument('--pii_annotator', type=str, required=True, help='PII detection model to use')
    parser.add_argument('--pii_threshold', type=float, default=None, help='Threshold for PII detection')
    return parser, ['pii_annotator', 'pii_threshold']

def load_data(data_kwargs: Optional[dict]) -> List[DatasetRecord]:
    data = data_kwargs.get("data")
    adapter = ADAPTER_REGISTRY.get(data)
    if adapter is None:
        raise ValueError(f"Adapter '{data}' not found.")

    dataset = adapter(**data_kwargs)
    return dataset
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load and Evaluate Anonymization Models")
    parser, _ = add_data_args(parser)
    parser, _ = add_model_args(parser)
    
    args = parser.parse_args()

    adapter = ADAPTER_REGISTRY[args.data]
    dataset, train_dataset, val_dataset, test_dataset = None, None, None, None
    if os.path.isdir(args.data_in):
        train_dataset = list(adapter(data_in=f"{args.data_in}/train.json"))
        val_dataset = list(adapter(data_in=f"{args.data_in}/dev.json"))
        test_dataset = list(adapter(data_in=f"{args.data_in}/test.json"))
    else:
        dataset = list(adapter(data_in=args.data_in))

    print(f"  Train: {len(train_dataset)} records")
    print(f"  Val: {len(val_dataset)} records")
    print(f"  Test: {len(test_dataset)} records")

    detector = PIIDetector(model_name=args.pii_annotator)

    detector.set_train_dataset(train_dataset)
    detector.set_val_dataset(val_dataset)
    detector.set_test_dataset(test_dataset)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    detector.train(
        epochs=3,
        output_dir=f"models/pii_detectors/{timestamp}",
        batch_size=4,
    )

    predictions = detector.predict(test_dataset)
    
    print(f"\n Predictions ({len(predictions)} records):")
    for pred in predictions:
        print(f"\n  Record: {pred.uid}")
        print(f"  Text: {pred.text}")
        if pred.spans:
            print(f"  Detected spans:")
            for span in pred.spans:
                print(f"    - [{span.start}:{span.end}] {span.text!r} -> {span.label} (conf: {span.confidence:.2f})")
        else:
            print(f"  No spans detected")
    
    print("\n Testing evaluate() method...")
    metrics = detector.evaluate(test_dataset)

    print("\n Evaluation Metrics:")
    for key, value in metrics.items():
        if key != "detailed_report":
            print(f"  {key}: {value}")
    print(" Detailed Report:")
    print(metrics["detailed_report"])