import argparse
import os
import json
from typing import List

from dp.loaders import ATTACKER_ADAPTER_REGISTRY, get_attacker_adapter
from dp.loaders.base import AttackerDatasetRecord

available_datasets = list(ATTACKER_ADAPTER_REGISTRY.keys())


def add_data_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument('--data', type=str, required=True, choices=available_datasets, help='Dataset name ({})'.format(", ".join(available_datasets)))
    parser.add_argument('--data_in', type=str, required=True, help='Path to input data file or directory')
    parser.add_argument('--max_records', type=int, default=None, help='Maximum number of records to load')


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate/load attacker record extensions (BK + summary)")
    add_data_args(parser)
    parser.add_argument('--full_record', action='store_true', help='Print full record details')
    parser.add_argument('--save_to_jsonl', type=str, help='Path to save processed extensions (JSONL)')
    parser.add_argument('--load_from_jsonl', type=str, help='Path to load processed extensions (JSONL)')

    args = parser.parse_args()

    adapter = get_attacker_adapter(args.data, data=args.data, data_in=args.data_in, max_records=args.max_records)

    if args.load_from_jsonl and os.path.exists(args.load_from_jsonl):
        print(f"Loading record extensions from {args.load_from_jsonl}...")
        adapter.load_cache_from_jsonl(args.load_from_jsonl)
        print("✓ Loaded precomputed extensions")

    # Truncate output file if requested
    if args.save_to_jsonl:
        os.makedirs(os.path.dirname(args.save_to_jsonl), exist_ok=True)
        with open(args.save_to_jsonl, 'w', encoding='utf-8') as f:
            pass

    unique_uids = set()
    unique_names = set()

    for record in adapter.iter_records(progress=True):
        if args.full_record:
            print(record)
        if args.save_to_jsonl:
            with open(args.save_to_jsonl, 'a', encoding='utf-8') as f:
                json_record = {
                    'uid': record.uid,
                    'background_knowledge': record.background_knowledge,
                    'summarized_text': record.summarized_text,
                }
                f.write(json.dumps(json_record) + '\n')
        unique_names.add(record.name)
        unique_uids.add(record.uid)

    print(f"Total individuals loaded: {len(unique_names)}")
    print(f"Total records loaded: {len(unique_uids)}")


if __name__ == "__main__":
    main()