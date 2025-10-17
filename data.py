from dp.loaders import ADAPTER_REGISTRY
import argparse
import os

available_datasets = list(ADAPTER_REGISTRY.keys())

def load_data(adapter_name: str, data_path: str, max_records: int = None):
    adapter = ADAPTER_REGISTRY.get(adapter_name)
    if not adapter:
        raise ValueError(f"Adapter '{adapter_name}' not found.")
    dataset = adapter(path=data_path, max_records=max_records)
    return dataset

def add_data_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument('--data', type=str, required=True, choices=available_datasets, help='Dataset name (trustpilot, tab, db_bio)')
    parser.add_argument('--data_in', type=str, required=True, help='Path to input data file or directory')
    parser.add_argument('--max_records', type=int, default=1, help='Maximum number of records to load')
    return parser

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark Anonymization Tools")
    parser = add_data_args(parser)

    args = parser.parse_args()

    dataset = load_data(args.data, args.data_in, args.max_records)

    for record in dataset:
        print(f"ID: {record.uid}\nText: {record.text[:100]}...\nAnnotations: {record.spans}\nUtilities: {record.utilities}\n")