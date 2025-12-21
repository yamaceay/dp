import argparse
import os
import json

from dp.tri.loaders import ATTACKER_ADAPTER_REGISTRY, get_attacker_adapter
from dp.loaders import read_batch_annotations_from_path
from dp.loaders.base import TextAnnotation

available_datasets = list(ATTACKER_ADAPTER_REGISTRY.keys())


def add_data_args(parser: argparse.ArgumentParser) -> list[str]:
    parser.add_argument('--data', type=str, required=True, choices=available_datasets, help='Dataset name ({})'.format(", ".join(available_datasets)))
    parser.add_argument('--data_in', type=str, required=True, help='Path to input data file or directory')
    parser.add_argument('--start', type=int, default=None, help='Start index for slicing (inclusive, python slicing semantics)')
    parser.add_argument('--end', type=int, default=None, help='End index for slicing (exclusive, python slicing semantics)')
    parser.add_argument('--step', type=int, default=None, help='Step for slicing (python slicing semantics)')
    parser.add_argument('--max_records', type=int, default=None, help='Maximum number of records to load after slicing')
    return ['data', 'data_in', 'start', 'end', 'step', 'max_records']


def _load_starting_anonymizations_by_idx(paths: list[str]) -> list[list[TextAnnotation]]:
    merged: list[list[TextAnnotation]] = []
    for path in paths:
        batch = read_batch_annotations_from_path(path)
        if not batch:
            continue
        if len(batch) > len(merged):
            merged.extend([[] for _ in range(len(batch) - len(merged))])
        for idx, annots in enumerate(batch):
            if annots:
                merged[idx].extend(annots)
    return merged

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate/load attacker record extensions (BK + summary)")
    data_keys = add_data_args(parser)
    parser.add_argument('--starting_anonymizations', type=str, nargs='*', default=None, help='Paths to starting anonymizations (JSONL)')
    parser.add_argument(
        '--starting_replacement',
        type=str,
        default=None,
        help='Optional explicit replacement token; if omitted, uses per-span replacement or label-derived token',
    )
    parser.add_argument('--full_record', action='store_true', help='Print full record details')
    parser.add_argument('--save_to_jsonl', type=str, help='Path to save processed extensions (JSONL)')
    parser.add_argument('--load_from_jsonl', type=str, help='Path to load processed extensions (JSONL)')

    args = parser.parse_args()

    data_kwargs = {k: getattr(args, k) for k in data_keys}
    adapter = get_attacker_adapter(data_kwargs.pop("data"), **data_kwargs)

    if args.starting_anonymizations:
        starting_by_idx = _load_starting_anonymizations_by_idx(args.starting_anonymizations)
        if hasattr(adapter, "set_starting_anonymizations"):
            adapter.set_starting_anonymizations(starting_by_idx, replacement=args.starting_replacement)
        else:
            raise ValueError("Selected attacker adapter does not support starting anonymizations")

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
