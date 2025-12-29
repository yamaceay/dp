from typing import Dict, List
from dp.loaders import ADAPTER_REGISTRY
from dp.experiments.utility.getters import UTILITY_TARGETS
import argparse

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

def add_data_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument('--data', type=str, required=True, choices=available_datasets, help='Dataset name ({})'.format(", ".join(available_datasets)))
    parser.add_argument('--data_in', type=str, required=True, help='Path to input data file or directory')
    parser.add_argument('--start', type=int, default=None, help='Start index for slicing (inclusive, python slicing semantics)')
    parser.add_argument('--end', type=int, default=None, help='End index for slicing (exclusive, python slicing semantics)')
    parser.add_argument('--step', type=int, default=None, help='Step for slicing (python slicing semantics)')
    parser.add_argument('--max_records', type=int, default=None, help='Maximum number of records to load after slicing')
    return ['data', 'data_in', 'start', 'end', 'step', 'max_records']

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark Anonymization Tools")
    data_keys = add_data_args(parser)
    parser.add_argument('--full_record', action='store_true', help='Print full record details')

    args = parser.parse_args()
    data_kwargs = {k: getattr(args, k) for k in data_keys}

    dataset = load_data(data_kwargs)

    value_getters = {
        'uid': lambda r: r.uid,
        'name': lambda r: r.name,
        'key': lambda r: list(r.metadata.keys()),
    }

    special_value_getters = UTILITY_TARGETS.get(args.data, {})
    for key, target in special_value_getters.items():
        value_getters[key] = target.getter

    print(value_getters)

    unique_values = {}
    sum_text_length = 0
    max_text_length = 0
    all_text_lengths = []

    for record in dataset.iter_records():
        if args.full_record:
            print(record)
        all_text_lengths.append(len(record.text))
        sum_text_length += len(record.text)
        max_text_length = max(max_text_length, len(record.text))

        for key, getter in value_getters.items():
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
    table_str = format_table(unique_key_list, unique_count_list, unique_value_list)
    print(table_str)
    print(f"Average text length: {sum_text_length / len(all_text_lengths)}" if dataset else "No records found.")
    print(f"Maximum text length: {max_text_length}" if dataset else "No records found.")
    print(f"All text lengths: {all_text_lengths}" if dataset else "No records found.")
