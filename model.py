from typing import Any, Dict, Optional, List, Tuple, Union
import argparse
import json
import yaml
import time
import itertools
from datetime import datetime

from dp.methods.anonymizer import Anonymizer, AnonymizationBuilder
from dp.methods.registry import MODEL_REGISTRY
from dp.methods.constants import (
    get_capabilities,
    PII_CLASSIFIER_MODEL_LIST,
    RISK_MASKER_MODEL_LIST,
)
from dp.loaders import (
    ADAPTER_REGISTRY,
    DatasetRecord,
    read_annotations,
    read_batch_annotations,
    read_batch_annotations_from_path,
    list_batch_timestamps,
)

from dp.utils.pii_detector import PIIDetector
from dp.utils.selector import PIIOnlySelector, AllSelector, ByRiskSelector
from dp.utils.explainer import UniformExplainer, GreedyExplainer, ShapExplainer
from dp.utils.chunking import TruncateChunker, SlidingWindowChunker, TokenAwareChunker
from dp.utils.output import OUTPUT_HANDLER_REGISTRY
from runtime import load_runtime_bundle

available_models = list(MODEL_REGISTRY.keys())
available_datasets = list(ADAPTER_REGISTRY.keys())
pii_confidence_models = set(PII_CLASSIFIER_MODEL_LIST)
risk_tolerance_models = set(RISK_MASKER_MODEL_LIST)

def add_data_args(parser: argparse.ArgumentParser) -> List[str]:
    parser.add_argument('--data', type=str, required=True, choices=available_datasets, help='Dataset name ({})'.format(", ".join(available_datasets)))
    parser.add_argument('--data_in', type=str, required=True, help='Path to input data file or directory')
    parser.add_argument('--start', type=int, default=None, help='Start index for slicing (inclusive, python slicing semantics)')
    parser.add_argument('--end', type=int, default=None, help='End index for slicing (exclusive, python slicing semantics)')
    parser.add_argument('--step', type=int, default=None, help='Step for slicing (python slicing semantics)')
    parser.add_argument('--max_records', type=int, default=None, help='Maximum number of records to load after slicing (default: None = all records)')
    return ['data', 'data_in', 'start', 'end', 'step', 'max_records']

def add_model_args(parser: argparse.ArgumentParser) -> List[str]:
    parser.add_argument('--model', type=str, required=True, choices=available_models, help='Anonymization model/method to evaluate')
    parser.add_argument('--model_in', type=str, default=None, help='Path to the method configuration')
    return ['model', 'model_in']

def add_runtime_args(parser: argparse.ArgumentParser) -> List[str]:
    parser.add_argument('--runtime_in', type=str, nargs='+', default=None, help='Path(s) to the runtime configuration(s)')
    parser.add_argument('--texts', type=str, nargs='+', help='Texts to anonymize (space-separated)')
    parser.add_argument('--indices', type=int, nargs='+', help='Indices of records to anonymize from dataset (space-separated)')
    parser.add_argument('--output', type=str, default='print', choices=list(OUTPUT_HANDLER_REGISTRY.keys()), help='Output handler type')
    parser.add_argument('--timestamp', type=str, default=None, help='Batch timestamp to use for output files (default: now)')
    parser.add_argument('--annotations', type=str, choices=['spacy', 'presidio', 'manual'], default=None, help='Type of starting annotations relevant for data preprocessing')
    parser.add_argument('--annotations_in', type=str, default=None, metavar='SOURCES', help='Load annotations from previous run (format: path/to/file.jsonl, comma-separated for multiple sources)')
    parser.add_argument('--list_annotations', action='store_true', help='List available annotation files and exit')
    parser.add_argument('--stream', action='store_true', help='Stream outputs (recommended for jsonl) instead of buffering all results')
    return ['runtime_in', 'texts', 'indices', 'output', 'annotations_in', 'list_annotations', 'stream']

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


def load_precomputed_risk(path: str) -> Dict[str, Dict[str, object]]:
    risk_map: Dict[str, Dict[str, object]] = {}
    with open(path, "r", encoding="utf-8") as reader:
        for line in reader:
            entry = line.strip()
            if not entry:
                continue
            payload = json.loads(entry)
            uid = payload.get("uid")
            if not uid:
                continue
            offsets = payload.get("offsets")
            scores = payload.get("scores")
            if offsets is None or scores is None:
                continue
            risk_map[str(uid)] = {
                "offsets": offsets,
                "scores": scores,
            }
    return risk_map

def load_model(model_config: Optional[dict], model_kwargs: Optional[dict], data_kwargs: Optional[dict], dataset: Optional[Any] = None) -> Anonymizer:
    model = model_kwargs.get("model")
    model_cls = MODEL_REGISTRY.get(model)
    if model_cls is None:
        raise ValueError(f"Model '{model}' not found.")

    capabilities = get_capabilities(model)
    
    if capabilities.must_use_dataset:
        if dataset is None:
            raise ValueError(f"{model} requires dataset to be loaded")
        if isinstance(dataset, list):
            dataset_records = dataset
        elif hasattr(dataset, "iter_records"):
            dataset_records = list(dataset.iter_records())
        else:
            dataset_records = list(dataset)
        model_instance = model_cls(**model_config, **model_kwargs, **data_kwargs)
        model_instance.add_dataset_records(dataset_records)
    else:
        model_instance = model_cls(**model_config, **model_kwargs, **data_kwargs)

    return model_instance

def compute_dataset_indices(dataset, data_kwargs: Dict[str, Any], record_count: int) -> List[int]:
    total = len(dataset)
    start = data_kwargs.get("start")
    start_value = 0 if start is None else start
    end = data_kwargs.get("end")
    stop_value = total if end is None else min(end, total)
    step = data_kwargs.get("step")
    step_value = 1 if step is None else step
    iterator = itertools.islice(range(total), start_value, stop_value, step_value)
    max_records = data_kwargs.get("max_records")
    if max_records is not None:
        iterator = itertools.islice(iterator, max_records)
    indices = list(iterator)
    if len(indices) != record_count:
        raise ValueError(f"Loaded {record_count} records but computed {len(indices)} dataset indices. Verify adapter slicing options.")
    return indices

def resolve_requested_indices(available: List[int], requested: Optional[List[int]]) -> List[int]:
    if requested is None:
        return list(available)
    available_set = set(available)
    missing = [idx for idx in requested if idx not in available_set]
    if missing:
        raise ValueError(f"Requested indices {missing} are not available in the loaded dataset slice.")
    return list(requested)

def build_dataset_selection(
    *,
    available_indices: List[int],
    index_lookup: Dict[int, int],
    requested_indices: Optional[List[int]],
) -> Tuple[List[int], List[int]]:
    selected = resolve_requested_indices(available_indices, requested_indices)
    dataset_indices = [index_lookup[idx] for idx in selected]
    return dataset_indices, selected

def build_text_selection(
    *,
    records: List[DatasetRecord],
    available_indices: List[int],
    index_lookup: Dict[int, int],
    requested_indices: Optional[List[int]],
    provided_texts: Optional[List[str]],
) -> Tuple[List[str], List[int], Optional[List[str]]]:
    if provided_texts is not None:
        texts = list(provided_texts)
        record_indices = list(range(len(texts)))
        record_names: Optional[List[str]] = None
    else:
        selected = resolve_requested_indices(available_indices, requested_indices)
        selected_records = [records[index_lookup[idx]] for idx in selected]
        texts = [record.text for record in selected_records]
        record_indices = selected
        record_names = [
            str(
                getattr(record, "uid", None)
                or getattr(record, "name", None)
                or idx
            )
            for record, idx in zip(selected_records, selected)
        ]
    return texts, record_indices, record_names

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


def stream_anonymization(
    *,
    anonymizer: Anonymizer,
    builder: AnonymizationBuilder,
    capabilities,
    runtime_config: Dict[str, Any],
    dataset_name: str,
    model_name: str,
    output_handler,
    dataset_indices: Optional[List[int]],
    texts: Optional[List[str]],
    record_indices: Optional[List[int]],
    task_id: int,
) -> Tuple[int, float]:
    processed = 0
    run_start = time.time()

    if capabilities.requires_k:
        if dataset_indices is None or record_indices is None:
            raise ValueError("Streaming requires dataset indices for k-anonymization methods.")
        stream = builder.anonymize_stream(**runtime_config)
        for position, per_idx in enumerate(stream):
            if position >= len(record_indices):
                raise ValueError("Stream produced more results than record indices provided.")
            idx_value = record_indices[position]
            for results in per_idx.values():
                for result in results:
                    output_handler.output(result, idx=idx_value, dataset=dataset_name, model=model_name, task_id=task_id)
            processed += 1

    elif capabilities.must_use_dataset:
        if dataset_indices is None or record_indices is None:
            raise ValueError("Streaming requires dataset indices for dataset-based methods.")
        filtered_kwargs = dict(runtime_config)
        for abs_idx, local_idx in zip(record_indices, dataset_indices):
            result = anonymizer.anonymize_from_dataset(idx=local_idx, **filtered_kwargs)
            output_handler.output(result, idx=abs_idx, dataset=dataset_name, model=model_name, task_id=task_id)
            processed += 1

    elif capabilities.requires_epsilon:
        if texts is None or not texts:
            raise ValueError("Streaming requires texts for DP methods.")
        if record_indices is None or len(record_indices) != len(texts):
            raise ValueError("Record indices must align with texts for streaming.")
        stream = builder.anonymize_stream(**runtime_config)
        for position, per_text in enumerate(stream):
            if position >= len(record_indices):
                raise ValueError("Stream produced more results than record indices provided.")
            idx_value = record_indices[position]
            for results in per_text.values():
                for result in results:
                    output_handler.output(
                        result,
                        idx=idx_value,
                        dataset=dataset_name,
                        model=model_name,
                        task_id=task_id,
                    )
            processed += 1

    else:
        if texts is None or not texts:
            raise ValueError("Streaming requires texts for this method.")
        if record_indices is None or len(record_indices) != len(texts):
            raise ValueError("Record indices must align with texts for streaming.")
        filtered_kwargs = dict(runtime_config)
        param_name: Optional[str] = None
        values: List[Optional[float]] = [None]
        if capabilities.is_pii_classifier:
            param_name = "pii_confidence"
            values = builder.request.pii_confidences or [None]
        elif capabilities.is_risk_masker:
            param_name = "risk_tolerance"
            values = builder.request.risk_tolerances or [None]

        def normalize_parameter_list(entries: Optional[List[Optional[float]]]) -> List[Optional[float]]:
            if not entries:
                return [None]
            ordered: List[Optional[float]] = []
            seen: set = set()
            for entry in entries:
                key = entry
                if key in seen:
                    continue
                seen.add(key)
                ordered.append(entry)
            return ordered

        def attach_metadata(result, key: Optional[str], value: Optional[float]) -> None:
            if key is None or value is None:
                return
            payload = dict(result.metadata or {})
            payload[key] = value
            result.metadata = payload

        ordered_values = normalize_parameter_list(values)
        setter = getattr(anonymizer, f"set_{param_name}", None) if param_name else None
        if setter is None and param_name is not None and any(value is not None for value in ordered_values):
            raise ValueError(f"{model_name} does not support '{param_name}' overrides")

        for value in ordered_values:
            if setter is not None and value is not None:
                setter(value)
            stream = anonymizer.anonymize_stream(texts=texts, **filtered_kwargs)
            for pos, result in enumerate(stream):
                if pos >= len(record_indices):
                    raise ValueError("Stream produced more results than record indices provided.")
                idx_value = record_indices[pos]
                attach_metadata(result, param_name, value)
                output_handler.output(
                    result,
                    idx=idx_value,
                    dataset=dataset_name,
                    model=model_name,
                    task_id=task_id,
                )
                processed += 1

    elapsed = time.time() - run_start
    return processed, elapsed
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch Anonymization - Process Multiple Records")
    data_keys = add_data_args(parser)
    model_keys = add_model_args(parser)
    runtime_keys = add_runtime_args(parser)
    
    args = parser.parse_args()
    data_kwargs = {k: getattr(args, k) for k in data_keys}
    model_kwargs = {k: getattr(args, k) for k in model_keys}
    runtime_kwargs = {k: getattr(args, k) for k in runtime_keys}
    runtime_inputs = runtime_kwargs.pop("runtime_in", None)
    runtime_bundle = load_runtime_bundle(runtime_inputs)
    runtime_config = dict(runtime_bundle.base_config)
    texts_arg = runtime_kwargs.pop("texts", None)
    indices_arg = runtime_kwargs.pop("indices", None)
    stream_enabled = runtime_kwargs.pop("stream", False)

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
    records = list(dataset.iter_records())

    model_config = load_config(args.model_in)
    if model_config is None:
        model_config = {}

    explainer_block = model_config.pop("explainer", None)

    explainer_name = None
    explainer_path = None
    risk_path = None
    risk_temperature = None
    if isinstance(explainer_block, dict):
        explainer_name = explainer_block.get("name")
        explainer_path = explainer_block.get("tri_pipeline")
        risk_temperature = explainer_block.get("risk_temperature")
        nested_risk_path = explainer_block.get("risk_scores")
        if nested_risk_path is not None:
            risk_path = nested_risk_path

    model_config["risk_temperature"] = risk_temperature
    
    capabilities = get_capabilities(args.model)
    
    pii_chunking = model_config.get("pii_chunking", {})
    tri_chunking = model_config.get("tri_chunking", {})

    loaded_annotations = None
    if capabilities.can_use_annotations and args.annotations_in:
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
    
    model = load_model(model_config, model_kwargs, data_kwargs, records)
    
    if loaded_annotations is not None and hasattr(model, 'set_annotations'):
        model.set_annotations(loaded_annotations, name=args.annotations)

    if capabilities.can_use_filtering:
        selector = AllSelector()
        token_selection_config = model_config.get("token_selection", {})
        type_of_selector = token_selection_config.get("name", None)

        if type_of_selector is not None and type_of_selector == "pii_only":
            pii_annotator_path = token_selection_config.get("pii_annotator", None)
            if pii_annotator_path is None:
                raise ValueError("PIIOnlySelector requires 'pii_annotator' path in model configuration.")
            pii_threshold = token_selection_config.get("pii_threshold", None)
            
            pii_use_chunking = pii_chunking.get("enabled", False)
            pii_annotator = PIIDetector(model_name=pii_annotator_path, use_chunking=pii_use_chunking)
            selector = PIIOnlySelector(pii_detector=pii_annotator, threshold=pii_threshold)
        
        elif type_of_selector is not None and type_of_selector == "by_risk":
            risk_tolerance = token_selection_config.get("risk_tolerance")
            if risk_tolerance is None:
                raise ValueError("ByRiskSelector requires 'risk_tolerance'.")
            selector = ByRiskSelector(risk_tolerance=risk_tolerance)

        model.set_filtering_strategy(selector)

    if capabilities.can_use_scoring:
        explainability = explainer_name
        tri_use_chunking = tri_chunking.get("enabled", False)
        
        if explainability is None:
            explainability = "uniform"
        
        if capabilities.must_use_non_uniform_explainer:
            if explainability == "uniform":
                raise ValueError(f"{args.model} requires explainability to be 'greedy' or 'shap', not 'uniform'")
            if explainer_path is None:
                raise ValueError(f"{args.model} requires an explainer tri_pipeline to be set")
        
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

    runtime_explainer = runtime_config.pop("explainer", None)
    runtime_risk_path = None
    if isinstance(runtime_explainer, dict):
        runtime_risk_path = runtime_explainer.get("risk_scores")
    if risk_path is None and runtime_risk_path is not None:
        risk_path = runtime_risk_path
    
    if risk_path:
        risk_scores = load_precomputed_risk(risk_path)
        if hasattr(model, "set_risk_scores"):
            model.set_risk_scores(risk_scores, records=records)

    batch_timestamp = args.timestamp if args.timestamp else datetime.now().strftime("%Y%m%d_%H%M%S")
    
    output_handler_cls = OUTPUT_HANDLER_REGISTRY.get(args.output, OUTPUT_HANDLER_REGISTRY["print"])
    
    if args.output in ["jsonl"]:
        output_handler = output_handler_cls(timestamp=batch_timestamp)
    else:
        output_handler = output_handler_cls()
    
    dataset_indices_all = compute_dataset_indices(dataset, data_kwargs, len(records))
    index_lookup = {idx: pos for pos, idx in enumerate(dataset_indices_all)}

    builder = model.builder()

    if texts_arg is not None and indices_arg is not None:
        raise ValueError("Cannot specify both --texts and --indices")

    texts: Optional[List[str]] = None
    record_indices: Optional[List[int]] = None
    dataset_indices: Optional[List[int]] = None

    if capabilities.must_use_dataset:
        if texts_arg is not None:
            raise ValueError(f"{args.model} requires dataset records, cannot use --texts. Use --indices instead or omit both to process all records.")
        dataset_indices, record_indices = build_dataset_selection(
            available_indices=dataset_indices_all,
            index_lookup=index_lookup,
            requested_indices=indices_arg,
        )
        builder.with_indices(dataset_indices)
    else:
        texts, record_indices, record_names = build_text_selection(
            records=records,
            available_indices=dataset_indices_all,
            index_lookup=index_lookup,
            requested_indices=indices_arg,
            provided_texts=texts_arg,
        )
        builder.with_texts(texts)
        if record_names is not None:
            runtime_config["record_names"] = record_names

    if record_indices is None:
        raise ValueError("No records resolved for processing.")

    if args.model in pii_confidence_models and runtime_bundle.pii_confidence_values:
        builder.with_pii_confidences(runtime_bundle.pii_confidence_values)

    if args.model in risk_tolerance_models and runtime_bundle.risk_tolerance_values:
        builder.with_risk_tolerances(runtime_bundle.risk_tolerance_values)

    if capabilities.requires_k:
        ks = runtime_bundle.k_values
        if not ks:
            ks = [5]
        builder.with_ks(ks)

    if capabilities.requires_epsilon:
        epsilons = runtime_bundle.epsilon_values
        if not epsilons:
            epsilons = [100.0]
        builder.with_epsilons(epsilons)

    if stream_enabled:
        if args.output != "jsonl":
            raise ValueError("--stream is currently supported only with --output jsonl")
        if not capabilities.supports_streaming:
            raise ValueError(f"{args.model} does not support --stream mode")
        processed, total_time = stream_anonymization(
            anonymizer=model,
            builder=builder,
            capabilities=capabilities,
            runtime_config=runtime_config,
            dataset_name=args.data,
            model_name=args.model,
            output_handler=output_handler,
            dataset_indices=dataset_indices,
            texts=texts,
            record_indices=record_indices,
            task_id=args.start,
        )
    else:
        start_time = time.time()
        results = []
        for result in builder.anonymize(**runtime_config):
            results.append(result)
        end_time = time.time()
        total_time = end_time - start_time
        num_records = len(texts) if not capabilities.must_use_dataset else len(dataset_indices or [])
        flattened = flatten_results(results, record_indices)

        flat_res, flat_indices = flattened
        output_results(
            flat_res,
            flat_indices,
            output_handler,
            verbose=args.output not in ["jsonl"],
            dataset=args.data,
            model=args.model,
            task_id=args.start,
        )

        processed = num_records
    
    avg_time = total_time / processed if processed > 0 else 0
    print(f"\n{'='*80}")
    print(f"Anonymization Performance:")
    print(f"  Total time: {total_time:.2f}s")
    print(f"  Texts processed: {processed}")
    print(f"  Average time per text: {avg_time:.2f}s")
    print(f"  Throughput: {processed/total_time:.2f} texts/s" if total_time > 0 and processed > 0 else "  Throughput: N/A")
    print('='*80)
 
    if hasattr(output_handler, 'close'):
        output_handler.close()
