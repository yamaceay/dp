import argparse
import os
import json
from typing import Any, Dict, Iterable, Optional, Sequence

from dp.tri.loaders import ATTACKER_ADAPTER_REGISTRY, get_attacker_adapter
from dp.loaders.base import DatasetAdapter, DatasetRecord
from dp.loaders.results import build_dataset_from_results
from dp.tri.sensitive_selectors import get_sensitive_selector
from runtime.config_loader import _read_yaml
from dp.utils.tasking import resolve_task_id, apply_task_template

available_datasets = list(ATTACKER_ADAPTER_REGISTRY.keys())


def add_data_args(parser: argparse.ArgumentParser) -> list[str]:
    parser.add_argument('--data', type=str, default=None, choices=available_datasets, help='Dataset name ({})'.format(", ".join(available_datasets)))
    parser.add_argument('--data_in', type=str, default=None, help='Path to input data file or directory')
    parser.add_argument('--split', type=str, default=None, help='Split file name/path under indices/{dataset}/...')
    parser.add_argument('--start', type=int, default=None, help='Start index for slicing (inclusive, python slicing semantics)')
    parser.add_argument('--end', type=int, default=None, help='End index for slicing (exclusive, python slicing semantics)')
    parser.add_argument('--step', type=int, default=None, help='Step for slicing (python slicing semantics)')
    parser.add_argument('--max_records', type=int, default=None, help='Maximum number of records to load after slicing')
    parser.add_argument('--deidentify', action='store_true', default=None, help='Apply de-identification to train background entries')
    return ['data', 'data_in', 'split', 'start', 'end', 'step', 'max_records']


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
    split = params.get('split')
    max_records = params.get('max_records')
    result_in = params.get('result_in')
    full_record = bool(params.get('full_record') or False)
    save_to_jsonl = params.get('save_to_jsonl')
    load_from_jsonl = params.get('load_from_jsonl')
    deidentify = bool(params.get('deidentify') or False)
    sensitive_selector = params.get('sensitive_selector')
    sensitive_selector_kwargs = params.get('sensitive_selector_kwargs') or {}
    n_train_samples_for_selected = params.get('n_train_samples_for_selected')
    n_train_samples_for_other = params.get('n_train_samples_for_other')
    n_eval_samples_for_selected = params.get('n_eval_samples_for_selected')
    n_eval_samples_for_other = params.get('n_eval_samples_for_other')
    task_id = params.get('task_id')
    if task_id is not None:
        if not isinstance(task_id, int) or task_id < 0:
            raise ValueError('task_id must be a non-negative integer when provided')
    if sensitive_selector is not None and not isinstance(sensitive_selector, str):
        raise ValueError('sensitive_selector must be a string when provided')
    if not isinstance(sensitive_selector_kwargs, dict):
        raise ValueError('sensitive_selector_kwargs must be a mapping')
    for key, value in {
        'n_train_samples_for_selected': n_train_samples_for_selected,
        'n_train_samples_for_other': n_train_samples_for_other,
        'n_eval_samples_for_selected': n_eval_samples_for_selected,
        'n_eval_samples_for_other': n_eval_samples_for_other,
    }.items():
        if value is not None and (not isinstance(value, int) or value <= 0):
            raise ValueError(f'{key} must be a positive integer when provided')
    return {
        'data': data,
        'data_in': data_in,
        'start': start,
        'end': end,
        'step': step,
        'split': split,
        'max_records': max_records,
        'result_in': result_in,
        'full_record': full_record,
        'save_to_jsonl': save_to_jsonl,
        'load_from_jsonl': load_from_jsonl,
        'deidentify': deidentify,
        'sensitive_selector': sensitive_selector,
        'sensitive_selector_kwargs': sensitive_selector_kwargs,
        'n_train_samples_for_selected': n_train_samples_for_selected,
        'n_train_samples_for_other': n_train_samples_for_other,
        'n_eval_samples_for_selected': n_eval_samples_for_selected,
        'n_eval_samples_for_other': n_eval_samples_for_other,
        'task_id': task_id,
    }


def _resolve_sensitive_names(
    dataset: str,
    records: Sequence[DatasetRecord],
    selector_name: Optional[str],
    selector_kwargs: Dict[str, Any],
) -> Optional[set[str]]:
    if selector_name is None:
        return None
    resolved_selector_name = str(selector_name).strip().lower()
    if resolved_selector_name == "auto":
        resolved_selector_name = dataset
    selector = get_sensitive_selector(resolved_selector_name)
    total_individuals = sorted({str(record.name).strip() for record in records if str(record.name).strip()})
    selected_names = {str(name).strip() for name in selector.select_individuals(total_individuals, **selector_kwargs) if str(name).strip()}
    if not selected_names:
        raise ValueError("Sensitive selector returned an empty set")
    print(
        f"sensitive selector='{resolved_selector_name}' selected {len(selected_names)} "
        f"of {len(total_individuals)} individuals"
    )
    return selected_names


def _cap_texts(values: list[str], cap: Optional[int]) -> list[str]:
    if cap is None:
        return list(values)
    return list(values[:cap])

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate/load attacker record extensions (BK + summary)")
    parser.add_argument('--config', type=str, default=None)
    data_keys = add_data_args(parser)
    parser.add_argument('--result_in', type=str, default=None, help='Path to anonymization results (JSONL)')
    parser.add_argument('--full_record', action='store_true', default=None, help='Print full record details')
    parser.add_argument('--save_to_jsonl', type=str, default=None, help='Path to save processed extensions (JSONL)')
    parser.add_argument('--load_from_jsonl', type=str, default=None, help='Path to load processed extensions (JSONL)')
    parser.add_argument('--task_id', type=int, default=None, help='Task id for task-aware path templates')

    raw = parser.parse_args()
    cfg = _read_yaml(raw.config) if raw.config else {}
    params = _merge_args(cfg, vars(raw))
    resolved = _resolve_params(params)
    task_id = resolve_task_id(resolved.get('task_id'))
    for key in ('data_in', 'split', 'result_in', 'save_to_jsonl', 'load_from_jsonl'):
        resolved[key] = apply_task_template(resolved.get(key), task_id)

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
    selected_names = _resolve_sensitive_names(
        dataset=dataset_name,
        records=records,
        selector_name=resolved['sensitive_selector'],
        selector_kwargs=resolved['sensitive_selector_kwargs'],
    )

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
        train_texts = list(record.train_texts or [])
        eval_texts = list(record.eval_texts or [])
        if selected_names is not None:
            is_selected = str(record.name).strip() in selected_names
            train_cap = resolved['n_train_samples_for_selected'] if is_selected else resolved['n_train_samples_for_other']
            eval_cap = resolved['n_eval_samples_for_selected'] if is_selected else resolved['n_eval_samples_for_other']
            train_texts = _cap_texts(train_texts, train_cap)
            eval_texts = _cap_texts(eval_texts, eval_cap)
        if resolved['full_record']:
            print(record)
        if resolved['save_to_jsonl']:
            with open(resolved['save_to_jsonl'], 'a', encoding='utf-8') as f:
                json_record = {
                    'name': record.name,
                    'train_texts': train_texts,
                    'eval_texts': eval_texts,
                    'test_texts': list(record.test_texts or []),
                }
                f.write(json.dumps(json_record) + '\n')
        unique_names.add(record.name)

    print(f"Total individuals loaded: {len(unique_names)}")
    print(f"Total records loaded: {len(unique_names)}")


if __name__ == "__main__":
    main()
