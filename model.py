from typing import Any, Dict, Optional, List, Tuple, Union
import argparse
import yaml
import time
from datetime import datetime

from dp.methods.anonymizer import Anonymizer
from dp.methods.registry import MODEL_REGISTRY
from dp.methods.constants import get_capabilities
from dp.loaders import (
    ADAPTER_REGISTRY,
    DatasetRecord,
    read_annotations,
    read_batch_annotations,
    read_batch_annotations_from_path,
    list_batch_timestamps,
)

from dp.utils.pii_detector import PIIDetector
from dp.utils.selector.pii_only_selector import PIIOnlySelector
from dp.utils.explainer import UniformExplainer, GreedyExplainer, ShapExplainer
from dp.utils.chunking import TruncateChunker, SlidingWindowChunker, TokenAwareChunker
from dp.utils.output import OUTPUT_HANDLER_REGISTRY

available_models = list(MODEL_REGISTRY.keys())
available_datasets = list(ADAPTER_REGISTRY.keys())

def add_data_args(parser: argparse.ArgumentParser) -> List[str]:
    parser.add_argument('--data', type=str, required=True, choices=available_datasets, help='Dataset name ({})'.format(", ".join(available_datasets)))
    parser.add_argument('--data_in', type=str, required=True, help='Path to input data file or directory')
    parser.add_argument('--max_records', type=int, default=None, help='Maximum number of records to load (default: None = all records)')
    return ['data', 'data_in', 'max_records']

def add_model_args(parser: argparse.ArgumentParser) -> List[str]:
    parser.add_argument('--model', type=str, required=True, choices=available_models, help='Anonymization model/method to evaluate')
    parser.add_argument('--model_in', type=str, default=None, help='Path to the method configuration')
    return ['model', 'model_in']

def add_runtime_args(parser: argparse.ArgumentParser) -> List[str]:
    parser.add_argument('--runtime_in', type=str, default=None, help='Path to the runtime configuration')
    parser.add_argument('--texts', type=str, nargs='+', help='Texts to anonymize (space-separated)')
    parser.add_argument('--indices', type=int, nargs='+', help='Indices of records to anonymize from dataset (space-separated)')
    parser.add_argument('--output', type=str, default='print', choices=list(OUTPUT_HANDLER_REGISTRY.keys()), help='Output handler type')
    parser.add_argument('--annotations', type=str, choices=['spacy', 'presidio', 'manual'], default=None, help='Type of starting annotations relevant for data preprocessing')
    parser.add_argument('--annotations_in', type=str, default=None, metavar='SOURCES', help='Load annotations from previous run (format: path/to/file.jsonl, comma-separated for multiple sources)')
    parser.add_argument('--list_annotations', action='store_true', help='List available annotation files and exit')
    return ['runtime_in', 'texts', 'indices', 'output', 'annotations_in', 'list_annotations']

def load_config(sth_in: Optional[str]) -> dict:
    config = {}
    if sth_in is not None:
        with open(sth_in, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
    return config

def load_data(data_kwargs: Optional[dict]) -> List[DatasetRecord]:
    data = data_kwargs.get("data")
    adapter = ADAPTER_REGISTRY.get(data)
    if adapter is None:
        raise ValueError(f"Adapter '{data}' not found.")

    dataset = adapter(**data_kwargs)
    return dataset

def load_model(model_config: Optional[dict], model_kwargs: Optional[dict], data_kwargs: Optional[dict], dataset: Optional[List[DatasetRecord]] = None) -> Anonymizer:
    model = model_kwargs.get("model")
    model_cls = MODEL_REGISTRY.get(model)
    if model_cls is None:
        raise ValueError(f"Model '{model}' not found.")

    capabilities = get_capabilities(model)
    
    if capabilities.must_use_dataset:
        if dataset is None:
            raise ValueError(f"{model} requires dataset to be loaded")
        model_instance = model_cls(**model_config, **model_kwargs, **data_kwargs)
        model_instance.add_dataset_records(list(dataset.iter_records()))
    else:
        model_instance = model_cls(**model_config, **model_kwargs, **data_kwargs)
    
    return model_instance

def use_indices(model_name: str, runtime_kwargs: dict, data_kwargs: dict, length: int) -> bool:
    capabilities = get_capabilities(model_name)
    if not capabilities.must_use_dataset:
        return False
    
    indices = runtime_kwargs.get("indices")
    if indices is not None:
        for idx in indices:
            if idx < 0 or idx >= length:
                raise ValueError(f"Index {idx} is out of bounds for dataset of length {length}.")
        return True
    
    max_records = data_kwargs.get("max_records")
    if max_records is None:
        max_records = length
    runtime_kwargs["indices"] = list(range(min(max_records, length)))
    return True

def flatten_results(
    nested_results: Any,
    indices: List[int],
) -> Union[Tuple[List[Any], List[Optional[int]]], Dict[Any, Tuple[List[Any], List[Optional[int]]]]]:
    if isinstance(nested_results, dict):
        flattened: Dict[Any, Tuple[List[Any], List[Optional[int]]]] = {}
        for key, value in nested_results.items():
            flattened[key] = flatten_results(value, indices)
        return flattened

    if not isinstance(nested_results, list):
        idx = indices[0] if indices else None
        return [nested_results], [idx]

    if not nested_results:
        return [], []

    first = nested_results[0]
    if not isinstance(first, list):
        flat_indices = indices if len(indices) == len(nested_results) else [None] * len(nested_results)
        return nested_results, flat_indices

    flat_results: List[Any] = []
    flat_indices: List[Optional[int]] = []
    for idx, idx_results in zip(indices, nested_results):
        if isinstance(idx_results, list):
            for result in idx_results:
                flat_results.append(result)
                flat_indices.append(idx)
        else:
            flat_results.append(idx_results)
            flat_indices.append(idx)
    return flat_results, flat_indices


def output_results(
    results: List,
    text_indices: List[Optional[int]],
    output_handler,
    verbose: bool,
    header: Optional[str] = None,
    **output_kwargs,
):
    total_results = len(results)
    if verbose and header:
        print(f"\n{'#'*80}")
        print(header)
        print(f"{'#'*80}")
    count = min(total_results, len(text_indices))
    for i in range(count):
        idx = text_indices[i]
        result = results[i]
        if verbose:
            print(f"\n{'='*80}")
            print(f"Result {i+1}/{total_results}")
            print('='*80)
        output_handler.output(result, idx=idx, **output_kwargs)
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch Anonymization - Process Multiple Records")
    data_keys = add_data_args(parser)
    model_keys = add_model_args(parser)
    runtime_keys = add_runtime_args(parser)
    
    args = parser.parse_args()
    data_kwargs = {k: getattr(args, k) for k in data_keys}
    model_kwargs = {k: getattr(args, k) for k in model_keys}
    runtime_kwargs = {k: getattr(args, k) for k in runtime_keys}

    if args.list_annotations:
        print(f"Available annotations for {args.data}/{args.model}:")
        timestamps = list_batch_timestamps(dataset=args.data, model=args.model)
        if not timestamps:
            print("  No annotation files found")
        else:
            for ts in timestamps:
                annotations = read_batch_annotations(args.data, args.model, ts)
                print(f"  - {ts} ({len(annotations)} records)")
        exit(0)

    dataset = load_data(data_kwargs)

    model_config = load_config(args.model_in)
    if model_config is None:
        model_config = {}
    
    capabilities = get_capabilities(args.model)
    
    pii_chunking = model_config.get("pii_chunking", {})
    tri_chunking = model_config.get("tri_chunking", {})

    loaded_annotations = None
    if capabilities.can_use_annotations and args.annotations_in:
        records = list(dataset.iter_records())
        loaded_annotations = {}
        
        for source in args.annotations_in.split(','):
            source = source.strip()
            print(f"Loading annotations from {source}")
            annotations_list = read_batch_annotations_from_path(source)
            
            for idx, annotations in enumerate(annotations_list):
                if annotations and idx < len(records):
                    uid = records[idx].uid if hasattr(records[idx], 'uid') else str(idx)
                    if uid not in loaded_annotations:
                        loaded_annotations[uid] = []
                    loaded_annotations[uid].extend(annotations)
        
        print(f"✓ Loaded annotations for {len(loaded_annotations)} records from {len(args.annotations_in.split(','))} source(s)")
    
    model = load_model(model_config, model_kwargs, data_kwargs, dataset)
    
    if loaded_annotations is not None and hasattr(model, 'set_annotations'):
        model.set_annotations(loaded_annotations, name=args.annotations)

    if capabilities.can_use_filtering:
        pii_annotator_path = model_config.get("pii_annotator", None)
        threshold = model_config.get("pii_threshold", None)
        pii_use_chunking = pii_chunking.get("enabled", False)

        if pii_annotator_path is not None:
            pii_annotator = PIIDetector(model_name=pii_annotator_path, use_chunking=pii_use_chunking)
            selector = PIIOnlySelector(pii_detector=pii_annotator, threshold=threshold)
            model.set_filtering_strategy(selector)

    if capabilities.can_use_scoring:
        explainer_path = model_config.get("explainer_path", None)
        
        explainability = model_config.get("explainability", None)
        tri_use_chunking = tri_chunking.get("enabled", False)
        
        if capabilities.must_use_non_uniform_explainer:
            if explainability is None or explainability == "uniform":
                raise ValueError(f"{args.model} requires explainability to be 'greedy' or 'shap', not 'uniform'")
            if explainer_path is None:
                raise ValueError(f"{args.model} requires explainer_path to be set")
        
        if explainability is None:
            explainability = "uniform"
        
        if explainability == "uniform":
            explainer = UniformExplainer()
            model.set_scoring_strategy(explainer)
        elif explainer_path is not None:
            if explainability == "greedy":
                explainer = GreedyExplainer(model_name=explainer_path, use_chunking=tri_use_chunking)
            elif explainability == "shap":
                explainer = ShapExplainer(model_name=explainer_path, use_chunking=tri_use_chunking)
            else:
                raise ValueError(f"Unknown explainability method: {explainability}")
            model.set_scoring_strategy(explainer)

    runtime_config = load_config(args.runtime_in)
    
    batch_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    output_handler_cls = OUTPUT_HANDLER_REGISTRY.get(args.output, OUTPUT_HANDLER_REGISTRY["print"])
    
    if args.output in ["jsonl"]:
        output_handler = output_handler_cls(timestamp=batch_timestamp)
    else:
        output_handler = output_handler_cls()
    
    texts = runtime_kwargs.get("texts")
    indices = runtime_kwargs.get("indices")
    
    if texts is not None and indices is not None:
        raise ValueError("Cannot specify both --texts and --indices")
    
    builder = model.builder()

    if capabilities.must_use_dataset:
        if texts is not None:
            raise ValueError(f"{args.model} requires dataset records, cannot use --texts. Use --indices instead or omit both to process all records.")
        
        if indices is None:
            max_records = data_kwargs.get("max_records", None)
            if max_records is None:
                max_records = len(dataset)
            indices = list(range(min(max_records, len(dataset))))
        else:
            for idx in indices:
                if idx < 0 or idx >= len(dataset):
                    raise ValueError(f"Index {idx} is out of bounds for dataset of length {len(dataset)}.")
        
        builder.with_indices(indices)
        text_indices = indices  # For dataset methods, use indices as text_indices
    
    else:
        if texts is None:
            if indices is not None:
                records = list(dataset.iter_records())
                texts = [records[idx].text for idx in indices]
                text_indices = indices
            else:
                max_records = data_kwargs.get("max_records", len(dataset))
                records = list(dataset.iter_records())[:max_records]
                texts = [record.text for record in records]
                text_indices = list(range(len(texts)))
        else:
            text_indices = [None] * len(texts)
        
        builder.with_texts(texts)

    if capabilities.requires_k:
        k = runtime_config.get("k", 5)
        if isinstance(k, list):
            if len(k) > 1:
                raise ValueError("Grid search over k values is not supported. Provide single k value.")
            k = k[0]
        
        builder.with_ks(k)

    if capabilities.requires_epsilon:
        epsilon = runtime_config.get("epsilon", 100.0)
        if isinstance(epsilon, list):
            if len(epsilon) > 1:
                raise ValueError("Grid search over epsilon values is not supported. Provide single epsilon value.")
            epsilon = epsilon[0]
        
        builder.with_epsilons(epsilon)

    start_time = time.time()
    results = builder.anonymize(**runtime_config)
    end_time = time.time()
    
    total_time = end_time - start_time
    num_texts = len(texts) if not capabilities.must_use_dataset else len(indices)
    avg_time = total_time / num_texts if num_texts > 0 else 0
    
    print(f"\n{'='*80}")
    print(f"Anonymization Performance:")
    print(f"  Total time: {total_time:.2f}s")
    print(f"  Texts processed: {num_texts}")
    print(f"  Average time per text: {avg_time:.2f}s")
    print(f"  Throughput: {num_texts/total_time:.2f} texts/s" if total_time > 0 else "  Throughput: N/A")
    print('='*80)

    flattened = flatten_results(results, text_indices)

    if isinstance(flattened, dict):
        for grid_value, (flat_res, flat_indices) in flattened.items():
            header = f"Grid value: {grid_value}"
            output_results(
                flat_res,
                flat_indices,
                output_handler,
                verbose=args.output not in ["jsonl"],
                header=header,
                dataset=args.data,
                model=args.model,
            )
    else:
        flat_res, flat_indices = flattened
        output_results(
            flat_res,
            flat_indices,
            output_handler,
            verbose=args.output not in ["jsonl"],
            dataset=args.data,
            model=args.model,
        )
 
    if hasattr(output_handler, 'close'):
        output_handler.close()
