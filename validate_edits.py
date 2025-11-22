from typing import Dict, List
import argparse
import json
import sys

from dp.loaders import ADAPTER_REGISTRY
from dp.utils.token_edits import apply_token_edits

available_datasets = list(ADAPTER_REGISTRY.keys())


def add_data_args(parser: argparse.ArgumentParser) -> List[str]:
    parser.add_argument('--data', type=str, required=True, choices=available_datasets)
    parser.add_argument('--data_in', type=str, required=True)
    parser.add_argument('--start', type=int, default=None, help='Start index for slicing (inclusive, python slicing semantics)')
    parser.add_argument('--end', type=int, default=None, help='End index for slicing (exclusive, python slicing semantics)')
    parser.add_argument('--step', type=int, default=None, help='Step for slicing (python slicing semantics)')
    parser.add_argument('--max_records', type=int, default=None, help='Maximum number of records to load after slicing')
    return ['data', 'data_in', 'start', 'end', 'step', 'max_records']


def add_validation_args(parser: argparse.ArgumentParser) -> List[str]:
    parser.add_argument('--annotations_in', type=str, required=True)
    return ['annotations_in']


def load_model_output(path: str) -> List[Dict[str, object]]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_data(data_kwargs: Dict[str, object]) -> List:
    data = data_kwargs.get("data")
    adapter = ADAPTER_REGISTRY.get(data)
    if adapter is None:
        raise ValueError(f"Adapter '{data}' not found.")
    dataset = adapter(**data_kwargs)
    return list(dataset.iter_records())


def validate_offset_consistency(
    model_output: List[Dict[str, object]],
    original_records: List,
) -> int:
    failed_count = 0
    passed_count = 0
    skipped_count = 0
    
    for index, (model_record, original_record) in enumerate(zip(model_output, original_records)):
        original_text = original_record.text
        anonymized_text = model_record.get("text")
        metadata = model_record.get("metadata", {})
        token_edits = metadata.get("token_edits")
        
        if anonymized_text is None or token_edits is None:
            skipped_count += 1
            continue
        
        rebuilt_text = apply_token_edits(original_text, token_edits)
        
        if rebuilt_text == anonymized_text:
            passed_count += 1
        else:
            failed_count += 1
            print(f"\n{'='*80}")
            print(f"FAILED: Record {index}")
            print('='*80)
            print(f"Original length: {len(original_text)}")
            print(f"Anonymized length: {len(anonymized_text)}")
            print(f"Rebuilt length: {len(rebuilt_text)}")
            print(f"\nFirst 200 chars of original:\n{original_text[:200]}")
            print(f"\nFirst 200 chars of anonymized:\n{anonymized_text[:200]}")
            print(f"\nFirst 200 chars of rebuilt:\n{rebuilt_text[:200]}")
    
    return failed_count


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    data_keys = add_data_args(parser)
    validation_keys = add_validation_args(parser)
    
    args = parser.parse_args()
    data_kwargs = {k: getattr(args, k) for k in data_keys}
    validation_kwargs = {k: getattr(args, k) for k in validation_keys}
    
    print("Loading original records...")
    original_records = load_data(data_kwargs)
    
    print("Loading model output...")
    model_output = load_model_output(validation_kwargs["annotations_in"])
    
    if len(model_output) != len(original_records):
        print(f"\n{'='*80}")
        print("ERROR: Record count mismatch")
        print('='*80)
        print(f"Model output records: {len(model_output)}")
        print(f"Original records: {len(original_records)}")
        print('='*80)
        sys.exit(1)
    
    print(f"\n{'='*80}")
    print("Validating offset consistency")
    print('='*80)
    print(f"Total records: {len(model_output)}")
    print('='*80)
    
    failed_count = validate_offset_consistency(model_output, original_records)
    
    print(f"\n{'='*80}")
    print("Validation Results")
    print('='*80)
    print(f"Total records: {len(model_output)}")
    print(f"Passed: {len(model_output) - failed_count}")
    print(f"Failed: {failed_count}")
    print('='*80)
    
    if failed_count > 0:
        sys.exit(1)
    else:
        print("\nAll records passed offset consistency validation.")
        sys.exit(0)
