import argparse
import os
import json
from typing import Any, Dict, Iterable

from dp.tri.loaders import ATTACKER_ADAPTER_REGISTRY, get_attacker_adapter
from dp.loaders.base import DatasetAdapter, DatasetRecord
from dp.loaders.results import build_dataset_from_results
from runtime.config_loader import _read_yaml

available_datasets = list(ATTACKER_ADAPTER_REGISTRY.keys())


def add_data_args(parser: argparse.ArgumentParser) -> list[str]:
    parser.add_argument('--data', type=str, default=None, choices=available_datasets, help='Dataset name ({})'.format(", ".join(available_datasets)))
    parser.add_argument('--data_in', type=str, default=None, help='Path to input data file or directory')
    parser.add_argument('--start', type=int, default=None, help='Start index for slicing (inclusive, python slicing semantics)')
    parser.add_argument('--end', type=int, default=None, help='End index for slicing (exclusive, python slicing semantics)')
    parser.add_argument('--step', type=int, default=None, help='Step for slicing (python slicing semantics)')
    parser.add_argument('--max_records', type=int, default=None, help='Maximum number of records to load after slicing')
    parser.add_argument('--deidentify', action='store_true', default=None, help='Apply de-identification to train background entries')
    return ['data', 'data_in', 'start', 'end', 'step', 'max_records']


class InMemoryDatasetAdapter(DatasetAdapter):
    def __init__(self, records: list[DatasetRecord]) -> None:
        self._records = records

    def iter_records(self, *args, **kwargs) -> Iterable[DatasetRecord]:
        return iter(self._records)

    def __len__(self) -> int:
        return len(self._records)

def _merge_args(config: Dict[str, Any], args: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(config or {})
    for k, v in args.items():
        if v is not None:
            merged[k] = v
    return merged

def _resolve_params(params: Dict[str, Any]) -> Dict[str, Any]:
    data = params.get('data')
    data_in = params.get('data_in')
    start = params.get('start')
    end = params.get('end')
    step = params.get('step')
    max_records = params.get('max_records')
    result_in = params.get('result_in')
    full_record = bool(params.get('full_record') or False)
    save_to_jsonl = params.get('save_to_jsonl')
    load_from_jsonl = params.get('load_from_jsonl')
    deidentify = bool(params.get('deidentify') or False)
    return {
        'data': data,
        'data_in': data_in,
        'start': start,
        'end': end,
        'step': step,
        'max_records': max_records,
        'result_in': result_in,
        'full_record': full_record,
        'save_to_jsonl': save_to_jsonl,
        'load_from_jsonl': load_from_jsonl,
        'deidentify': deidentify,
    }

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate/load attacker record extensions (BK + summary)")
    parser.add_argument('--config', type=str, default=None)
    data_keys = add_data_args(parser)
    parser.add_argument('--result_in', type=str, default=None, help='Path to anonymization results (JSONL)')
    parser.add_argument('--full_record', action='store_true', default=None, help='Print full record details')
    parser.add_argument('--save_to_jsonl', type=str, default=None, help='Path to save processed extensions (JSONL)')
    parser.add_argument('--load_from_jsonl', type=str, default=None, help='Path to load processed extensions (JSONL)')

    raw = parser.parse_args()
    cfg = _read_yaml(raw.config) if raw.config else {}
    params = _merge_args(cfg, vars(raw))
    resolved = _resolve_params(params)

    if not resolved['data'] or not resolved['data_in']:
        raise ValueError('data and data_in are required (pass via args or --config)')

    data_kwargs = {k: resolved[k] for k in data_keys}
    dataset_name = data_kwargs.pop("data")
    adapter_kwargs = dict(**data_kwargs)
    if dataset_name in {"reddit", "yelp"}:
        adapter_kwargs["need_to_deidentify"] = resolved['deidentify']
    adapter = get_attacker_adapter(dataset_name, **adapter_kwargs)

    original_records = list(adapter.adapter.iter_records())
    if resolved['result_in']:
        records, _ = build_dataset_from_results(
            resolved['result_in'], 
            original_records,
            start=resolved['start'],
            end=resolved['end'],
            step=resolved['step'],
            max_records=resolved['max_records']
        )
    else:
        records = original_records
    adapter.adapter = InMemoryDatasetAdapter(records)

    if resolved['load_from_jsonl'] and os.path.exists(resolved['load_from_jsonl']):
        print(f"Loading record extensions from {resolved['load_from_jsonl']}...")
        adapter.load_cache_from_jsonl(resolved['load_from_jsonl'])
        print("✓ Loaded precomputed extensions")

    if resolved['save_to_jsonl']:
        os.makedirs(os.path.dirname(resolved['save_to_jsonl']), exist_ok=True)
        with open(resolved['save_to_jsonl'], 'w', encoding='utf-8') as f:
            pass

    unique_names = set()

    for record in adapter.iter_records(progress=True):
        if resolved['full_record']:
            print(record)
        if resolved['save_to_jsonl']:
            with open(resolved['save_to_jsonl'], 'a', encoding='utf-8') as f:
                json_record = {
                    'name': record.name,
                    'train_texts': list(record.train_texts or []),
                    'eval_texts': list(record.eval_texts or []),
                }
                f.write(json.dumps(json_record) + '\n')
        unique_names.add(record.name)

    print(f"Total individuals loaded: {len(unique_names)}")
    print(f"Total records loaded: {len(unique_names)}")


if __name__ == "__main__":
    main()
