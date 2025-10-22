from dp.loaders import ADAPTER_REGISTRY
import argparse
import os

available_datasets = list(ADAPTER_REGISTRY.keys())

def load_data(data: str, data_in: str, max_records: int = None):
    adapter = ADAPTER_REGISTRY.get(data)
    if not adapter:
        raise ValueError(f"Adapter '{data}' not found.")
    dataset = adapter(data, data_in=data_in, max_records=max_records)
    return dataset

def add_data_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument('--data', type=str, required=True, choices=available_datasets, help='Dataset name ({})'.format(", ".join(available_datasets)))
    parser.add_argument('--data_in', type=str, required=True, help='Path to input data file or directory')
    parser.add_argument('--max_records', type=int, default=None, help='Maximum number of records to load')
    return ['data', 'data_in', 'max_records']

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark Anonymization Tools")
    add_data_args(parser)

    args = parser.parse_args()

    dataset = load_data(args.data, args.data_in, args.max_records)

    unique_uids = set()
    texts = []

    utility_keys = set()

    for record in dataset.iter_records():
        unique_uids.add(record.uid)
        texts.append(record.text[:100])
        utility_keys.update(record.utilities.keys())
        # print(f"ID: {record.uid}\n"
        #       f"Text: {record.text[:100]}...\n"
        #       f"Utilities: {record.utilities}\n")

    print(f"Total individuals loaded: {len(unique_uids)}")
    print(f"Total records loaded: {len(texts)}")
    print(f"Utility keys found: {', '.join(utility_keys)}")