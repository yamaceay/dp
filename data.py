from typing import Callable, Dict, List, Optional
from dp.loaders import ADAPTER_REGISTRY
from dp.loaders.derive import DERIVE_REGISTRY
import argparse
import re

available_datasets = list(ADAPTER_REGISTRY.keys())


def format_table(unique_keys: List[str], unique_counts: List[int], unique_values: List[str]) -> str:
    """Format data as a simple table string."""
    count_keys = [f"Unique {key}s[{count}]" for key, count in zip(unique_keys, unique_counts)]
    col_width = max(len(str(item)) for item in count_keys)
    padded_strs = [f"{count_key:<{col_width}} : {values}" for count_key, values in zip(count_keys, unique_values)]
    return "\n".join(padded_strs)

def load_data(data_kwargs: Dict[str, object]):
    data = data_kwargs.get("data")
    adapter = ADAPTER_REGISTRY.get(str(data) if data is not None else None)
    if not adapter:
        raise ValueError(f"Adapter '{data}' not found.")
    dataset = adapter(**data_kwargs)
    return dataset


def evaluate_where(
    record: object,
    value_getters: Dict[str, Callable[[object], object]],
    where: Optional[str],
) -> bool:
    if where is None or where.strip() == "":
        return True
    scope: Dict[str, object] = {}
    for key, getter in value_getters.items():
        scope[key] = getter(record)

    def re_search(pattern: str, value: object) -> bool:
        return re.search(pattern, str(value)) is not None

    def re_match(pattern: str, value: object) -> bool:
        return re.match(pattern, str(value)) is not None

    def re_fullmatch(pattern: str, value: object) -> bool:
        return re.fullmatch(pattern, str(value)) is not None

    try:
        return bool(eval(where, {"__builtins__": {}, "re_search": re_search, "re_match": re_match, "re_fullmatch": re_fullmatch}, scope))
    except Exception as exc:
        raise ValueError(f"Invalid --where expression: {where}") from exc

def add_data_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument('--data', type=str, required=True, choices=available_datasets, help='Dataset name ({})'.format(", ".join(available_datasets)))
    parser.add_argument('--data_in', type=str, required=True, help='Path to input data file or directory')
    parser.add_argument('--split', type=str, default=None, help='Split file name/path under indices/{dataset}/...')
    parser.add_argument('--start', type=int, default=None, help='Start index for slicing (inclusive, python slicing semantics)')
    parser.add_argument('--end', type=int, default=None, help='End index for slicing (exclusive, python slicing semantics)')
    parser.add_argument('--step', type=int, default=None, help='Step for slicing (python slicing semantics)')
    parser.add_argument('--where', type=str, default=None, help='Filter condition over value_getter keys, e.g. "split == \'train\'" or "re_search(\'^ab\', name)"')
    parser.add_argument('--select', type=str, default=None, help='Comma-separated list of value_getter keys to include in output (default: all)')
    parser.add_argument('--max_records', type=int, default=None, help='Maximum number of records to load after slicing')
    return ['data', 'data_in', 'split', 'start', 'end', 'step', 'max_records']


def parse_selected_getters(select_arg: Optional[str], value_getters: Dict[str, Callable[[object], object]]) -> Dict[str, Callable[[object], object]]:
    if select_arg is None or select_arg.strip() == "":
        return value_getters
    selected_keys = [key.strip() for key in select_arg.split(",") if key.strip() != ""]
    invalid_keys = [key for key in selected_keys if key not in value_getters]
    if invalid_keys:
        valid_keys = ", ".join(sorted(value_getters.keys()))
        invalid_keys_str = ", ".join(invalid_keys)
        raise ValueError(f"Invalid --select keys: {invalid_keys_str}. Available keys: {valid_keys}")
    return {key: value_getters[key] for key in selected_keys}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark Anonymization Tools")
    data_keys = add_data_args(parser)
    parser.add_argument('--full_record', action='store_true', help='Print full record details')
    parser.add_argument('--unique_counts', action='store_true', help='Print unique value counts for each field')
    parser.add_argument('--functional', action='store_true', help='Perform functional analysis on dataset')

    args = parser.parse_args()
    data_kwargs = {k: getattr(args, k) for k in data_keys}
    has_where = args.where is not None and args.where.strip() != ""
    if has_where:
        data_kwargs['start'] = None
        data_kwargs['end'] = None
        data_kwargs['step'] = None
        data_kwargs['max_records'] = None

    dataset = load_data(data_kwargs)

    value_getters = {
        'name': lambda r: r.name,
        'key': lambda r: list(r.metadata.keys()),
        'split': lambda r: r.metadata.get("split"),
    }

    special_value_getters = DERIVE_REGISTRY.get(args.data, {})
    for key, target in special_value_getters.items():
        value_getters[key] = target
    selected_value_getters = parse_selected_getters(args.select, value_getters)

    unique_values: Dict[str, Dict[object, int]] = {}
    sum_text_length = 0
    max_text_length = 0
    all_text_lengths = []

    if has_where:
        filtered_records = [record for record in dataset.iter_records() if evaluate_where(record, value_getters, args.where)]
        records_to_process = filtered_records[slice(args.start, args.end, args.step)]
        if args.max_records is not None:
            records_to_process = records_to_process[:args.max_records]
    else:
        records_to_process = dataset.iter_records()

    for record in records_to_process:
        if args.full_record:
            print(record)
        all_text_lengths.append(len(record.text))

        for key, getter in selected_value_getters.items():
            value = getter(record)
            if value is None:
                continue
            unique_value = unique_values.get(key, dict())
            not_a_list = not isinstance(value, (list, set))
            if not_a_list :
                value = [value]
            for v in value:
                unique_value[v] = unique_value.get(v, 0) + 1
            unique_values[key] = unique_value

    unique_key_list, unique_value_list, unique_count_list = [], [], []
    for key, values in unique_values.items():
        unique_key_list.append(key)
        unique_values = None
        if any(count > 1 for _, count in values.items()):
            values_sorted = sorted(values.items(), key=lambda item: item[1], reverse=True)
            unique_values = [f"{v}[{c}]" for v, c in values_sorted]
        else:
            unique_values = list(values.keys())
        unique_value_list.append(unique_values)
        unique_count_list.append(len(unique_values))
    
    if args.unique_counts:
        table_str = format_table(unique_key_list, unique_count_list, unique_value_list)
        print(table_str)
    print(f"Text length: avg={sum(all_text_lengths)/len(all_text_lengths):.2f}, max={max(all_text_lengths)}, min={min(all_text_lengths)}")
    print(f"Total records: {len(all_text_lengths)}")

    if args.functional:
        from dp.loaders.func import FunctionalAnalysis
        if has_where:
            functional_records = records_to_process
        else:
            functional_records = list(dataset.iter_records())
        exclude_keys = ["key"] if "key" in selected_value_getters else []
        func_analysis = FunctionalAnalysis(functional_records, selected_value_getters, exclude_keys=exclude_keys)
        func_analysis.analyze()
        functionals = func_analysis.dag()
        print("Functional mappings found:")
        for functional in functionals:
            print(functional.show(head_n=3, tail_n=3))
