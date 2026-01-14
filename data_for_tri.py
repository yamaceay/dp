import argparse
import os
import json
from typing import Iterable

from dp.tri.loaders import ATTACKER_ADAPTER_REGISTRY, get_attacker_adapter
from dp.loaders.base import DatasetAdapter, DatasetRecord
from dp.loaders.results import build_dataset_from_results

available_datasets = list(ATTACKER_ADAPTER_REGISTRY.keys())


def add_data_args(parser: argparse.ArgumentParser) -> list[str]:
    parser.add_argument('--data', type=str, required=True, choices=available_datasets, help='Dataset name ({})'.format(", ".join(available_datasets)))
    parser.add_argument('--data_in', type=str, required=True, help='Path to input data file or directory')
    parser.add_argument('--start', type=int, default=None, help='Start index for slicing (inclusive, python slicing semantics)')
    parser.add_argument('--end', type=int, default=None, help='End index for slicing (exclusive, python slicing semantics)')
    parser.add_argument('--step', type=int, default=None, help='Step for slicing (python slicing semantics)')
    parser.add_argument('--max_records', type=int, default=None, help='Maximum number of records to load after slicing')
    return ['data', 'data_in', 'start', 'end', 'step', 'max_records']


class InMemoryDatasetAdapter(DatasetAdapter):
    def __init__(self, records: list[DatasetRecord]) -> None:
        self._records = records

    def iter_records(self, *args, **kwargs) -> Iterable[DatasetRecord]:
        return iter(self._records)

    def __len__(self) -> int:
        return len(self._records)

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate/load attacker record extensions (BK + summary)")
    data_keys = add_data_args(parser)
    parser.add_argument('--result_in', type=str, default=None, help='Path to anonymization results (JSONL)')
    parser.add_argument('--full_record', action='store_true', help='Print full record details')
    parser.add_argument('--save_to_jsonl', type=str, help='Path to save processed extensions (JSONL)')
    parser.add_argument('--load_from_jsonl', type=str, help='Path to load processed extensions (JSONL)')

    args = parser.parse_args()

    data_kwargs = {k: getattr(args, k) for k in data_keys}
    adapter = get_attacker_adapter(data_kwargs.pop("data"), **data_kwargs)

    original_records = list(adapter.adapter.iter_records())
    if args.result_in:
        records, _ = build_dataset_from_results(
            args.result_in, 
            original_records,
            start=args.start,
            end=args.end,
            step=args.step,
            max_records=args.max_records
        )
    else:
        records = original_records
    adapter.adapter = InMemoryDatasetAdapter(records)
    if hasattr(adapter, "_build_persona_records"):
        adapter._persona_records = adapter._build_persona_records()

    if args.load_from_jsonl and os.path.exists(args.load_from_jsonl):
        print(f"Loading record extensions from {args.load_from_jsonl}...")
        adapter.load_cache_from_jsonl(args.load_from_jsonl)
        print("✓ Loaded precomputed extensions")

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
                    'rewrited_text': record.rewrited_text,
                }
                f.write(json.dumps(json_record) + '\n')
        unique_names.add(record.name)
        unique_uids.add(record.uid)

    print(f"Total individuals loaded: {len(unique_names)}")
    print(f"Total records loaded: {len(unique_uids)}")


if __name__ == "__main__":
    main()
