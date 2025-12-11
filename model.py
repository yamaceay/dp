from typing import Any, Dict, Optional, List, Tuple
import argparse
import json
import yaml
import time
from datetime import datetime

from dp.methods.anonymizer import Anonymizer, AnonymizationBuilder
from dp.methods.registry import MODEL_REGISTRY, get_capabilities
from dp.methods.constants import PII_CLASSIFIER_MODEL_LIST, RISK_MASKER_MODEL_LIST
from dp.loaders import ADAPTER_REGISTRY, DatasetRecord, read_batch_annotations, read_batch_annotations_from_path, list_batch_timestamps
from dp.utils.pii_detector import PIIDetector
from dp.utils.selector import PIIOnlySelector, AllSelector, ByRiskSelector, UntilKSelector
from dp.utils.explainer import UniformExplainer, GreedyExplainer, ShapExplainer
from dp.utils.output import OUTPUT_HANDLER_REGISTRY
from runtime import load_runtime_bundle

available_models = list(MODEL_REGISTRY.keys())
available_datasets = list(ADAPTER_REGISTRY.keys())


def add_data_args(parser: argparse.ArgumentParser) -> List[str]:
    parser.add_argument('--data', type=str, required=True, choices=available_datasets)
    parser.add_argument('--data_in', type=str, required=True)
    parser.add_argument('--start', type=int, default=None)
    parser.add_argument('--end', type=int, default=None)
    parser.add_argument('--step', type=int, default=None)
    parser.add_argument('--max_records', type=int, default=None)
    return ['data', 'data_in', 'start', 'end', 'step', 'max_records']


def add_model_args(parser: argparse.ArgumentParser) -> List[str]:
    parser.add_argument('--model', type=str, required=True, choices=available_models)
    parser.add_argument('--model_in', type=str, default=None)
    return ['model', 'model_in']


def add_runtime_args(parser: argparse.ArgumentParser) -> List[str]:
    parser.add_argument('--runtime_in', type=str, nargs='+', default=None)
    parser.add_argument('--texts', type=str, nargs='+')
    parser.add_argument('--indices', type=int, nargs='+')
    parser.add_argument('--output', type=str, default='print', choices=list(OUTPUT_HANDLER_REGISTRY.keys()))
    parser.add_argument('--timestamp', type=str, default=None)
    parser.add_argument('--annotations', type=str, choices=['spacy', 'presidio', 'manual'], default=None)
    parser.add_argument('--annotations_in', type=str, default=None)
    parser.add_argument('--list_annotations', action='store_true')
    parser.add_argument('--stream', action='store_true')
    parser.add_argument('--unique_name', type=str, default=None)
    return ['runtime_in', 'texts', 'indices', 'output', 'annotations_in', 'list_annotations', 'stream', 'unique_name']


def load_config(path: Optional[str]) -> dict:
    if path is None:
        return {}
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f) or {}


def load_data(data_kwargs: dict) -> List[DatasetRecord]:
    adapter = ADAPTER_REGISTRY[data_kwargs["data"]]
    return list(adapter(**data_kwargs).iter_records())


def load_precomputed_risk(path: str) -> Dict[str, Dict[str, object]]:
    risk_map: Dict[str, Dict[str, object]] = {}
    with open(path, "r", encoding="utf-8") as reader:
        for line in reader:
            entry = line.strip()
            if not entry:
                continue
            payload = json.loads(entry)
            uid = payload.get("uid")
            if uid and payload.get("offsets") and payload.get("scores"):
                risk_map[str(uid)] = {"offsets": payload["offsets"], "scores": payload["scores"]}
    return risk_map


def extract_explainer_config(model_config: dict) -> Dict[str, Any]:
    block = model_config.pop("explainer", {}) or {}
    return {
        "name": block.get("name", "uniform"),
        "tri_pipeline": block.get("tri_pipeline"),
        "risk_temperature": block.get("risk_temperature"),
    }


def extract_precompute_config(model_config: dict) -> Dict[str, Any]:
    block = model_config.pop("precomputation", {}) or {}
    return {
        "risk_scores": block.get("risk_scores"),
    }


def extract_selector_config(model_config: dict) -> Dict[str, Any]:
    return model_config.get("token_selection", {})


def extract_chunking_config(model_config: dict, kind: str) -> Dict[str, Any]:
    return model_config.get(f"{kind}_chunking", {})


def build_selector(selector_config: dict):
    selector_type = selector_config.get("name")
    
    if selector_type == "pii_only":
        pii_path = selector_config.get("pii_annotator")
        if not pii_path:
            raise ValueError("PIIOnlySelector requires 'pii_annotator' in config")
        pii_chunking = selector_config.get("pii_chunking", {}).get("enabled", False)
        detector = PIIDetector(model_name=pii_path, use_chunking=pii_chunking)
        return PIIOnlySelector(detector, threshold=selector_config.get("pii_threshold"))
    
    if selector_type == "by_risk":
        risk_tolerance = selector_config.get("risk_tolerance")
        if risk_tolerance is None:
            raise ValueError("ByRiskSelector requires 'risk_tolerance'")
        return ByRiskSelector(risk_tolerance=risk_tolerance)
    
    if selector_type == "until_k":
        k = selector_config.get("k")
        if k is None:
            raise ValueError("UntilKSelector requires 'k' in config")
        return UntilKSelector(k=k)
    
    return AllSelector()


def build_explainer(explainer_config: dict, model_config: dict, capabilities, model_name: str):
    explainer_name = explainer_config["name"]
    
    if capabilities.must_use_non_uniform_explainer and explainer_name == "uniform":
        raise ValueError(f"{model_name} requires non-uniform explainability")
    
    if explainer_name == "uniform":
        return UniformExplainer()
    
    tri_pipeline = explainer_config.get("tri_pipeline")
    if not tri_pipeline:
        raise ValueError(f"{model_name} requires tri_pipeline path")
    
    tri_chunking = extract_chunking_config(model_config, "tri").get("enabled", False)
    
    if explainer_name == "greedy":
        return GreedyExplainer(model_name=tri_pipeline, use_chunking=tri_chunking)
    
    if explainer_name == "shap":
        return ShapExplainer(model_name=tri_pipeline, use_chunking=tri_chunking)
    
    raise ValueError(f"Unknown explainability: {explainer_name}")


def configure_model(model: Anonymizer, model_config: dict, explainer_config: dict, runtime_bundle, capabilities, model_name: str, records: List[DatasetRecord]):
    if explainer_config.get("risk_temperature") is not None:
        model_config["risk_temperature"] = explainer_config["risk_temperature"]
    
    selector_config = extract_selector_config(model_config)
    
    if capabilities.must_use_pii_selector:
        if not selector_config or selector_config.get("name") != "pii_only":
            raise ValueError(f"{model_name} requires pii_only selector")
        model.set_filtering_strategy(build_selector(selector_config))
    elif capabilities.must_use_risk_selector:
        if not selector_config or selector_config.get("name") != "by_risk":
            raise ValueError(f"{model_name} requires by_risk selector")
        model.set_filtering_strategy(build_selector(selector_config))
    elif capabilities.must_use_k_selector:
        if not selector_config or selector_config.get("name") != "until_k":
            raise ValueError(f"{model_name} requires until_k selector")
        model.set_filtering_strategy(build_selector(selector_config))
    elif capabilities.can_use_pii_selector or capabilities.can_use_risk_selector or capabilities.can_use_k_selector:
        if selector_config:
            model.set_filtering_strategy(build_selector(selector_config))
    
    if capabilities.can_use_scoring:
        model.set_scoring_strategy(build_explainer(explainer_config, model_config, capabilities, model_name))
    


def load_annotations(annotations_in: str, records: List[DatasetRecord]) -> Dict[str, List]:
    loaded_annotations = {}
    for source in annotations_in.split(','):
        for idx, annotations in enumerate(read_batch_annotations_from_path(source.strip())):
            if annotations and idx < len(records):
                uid = str(records[idx].uid if hasattr(records[idx], 'uid') else idx)
                loaded_annotations.setdefault(uid, []).extend(annotations)
    return loaded_annotations


def compute_dataset_indices(dataset_len: int, data_kwargs: Dict[str, Any]) -> List[int]:
    start = data_kwargs.get("start") or 0
    end = min(data_kwargs.get("end") or dataset_len, dataset_len)
    step = data_kwargs.get("step") or 1
    indices = list(range(start, end, step))
    max_records = data_kwargs.get("max_records")
    if max_records is not None:
        indices = indices[:max_records]
    return indices


def resolve_requested_indices(available: List[int], requested: Optional[List[int]]) -> List[int]:
    if requested is None:
        return available
    available_set = set(available)
    missing = [idx for idx in requested if idx not in available_set]
    if missing:
        raise ValueError(f"Requested indices {missing} not in available dataset slice")
    return requested


def initialize_builder_params(anonymizer: Anonymizer, runtime_bundle):
    if runtime_bundle.k_values:
        from dp.methods.constants import KParams
        return [KParams(ks=runtime_bundle.k_values)]
    if runtime_bundle.pii_confidence_values:
        return [("pii_confidence", runtime_bundle.pii_confidence_values)]
    if runtime_bundle.risk_tolerance_values:
        return [("risk_tolerance", runtime_bundle.risk_tolerance_values)]
    if hasattr(runtime_bundle, 'epsilon_values') and runtime_bundle.epsilon_values:
        from dp.methods.constants import EpsilonParam
        return [EpsilonParam(epsilon=runtime_bundle.epsilon_values[0])]
    return []


def flatten_results(nested: Any, indices: List[int]) -> Tuple[List[Any], List[Optional[int]]]:
    if isinstance(nested, dict):
        return {k: flatten_results(v, indices) for k, v in nested.items()}
    
    if not isinstance(nested, list):
        return [nested], [indices[0] if indices else None]
    
    if not nested:
        return [], []
    
    if not isinstance(nested[0], list):
        return nested, indices if len(indices) == len(nested) else [None] * len(nested)
    
    flat_results, flat_indices = [], []
    for idx, item_list in zip(indices, nested):
        for item in (item_list if isinstance(item_list, list) else [item_list]):
            flat_results.append(item)
            flat_indices.append(idx)
    
    return flat_results, flat_indices


def output_results(results: List, indices: List[Optional[int]], output_handler, verbose: bool, **kwargs):
    for i, (result, idx) in enumerate(zip(results, indices)):
        if verbose:
            print(f"\n{'='*80}\nResult {i+1}/{len(results)}\n{'='*80}")
        output_handler.output(result, idx=idx, **kwargs)


def stream_anonymize(anonymizer: Anonymizer, builder: AnonymizationBuilder, capabilities, output_handler, runtime_config: dict, record_indices: List[int], dataset_indices: Optional[List[int]], texts: Optional[List[str]], runtime_bundle, **metadata) -> Tuple[int, float]:
    run_start = time.time()
    processed = 0
    
    if capabilities.must_use_dataset:
        for abs_idx, local_idx in zip(record_indices, dataset_indices):
            result = anonymizer.anonymize_from_dataset(idx=local_idx, **runtime_config)
            output_handler.output(result, idx=abs_idx, **metadata)
            processed += 1
    else:
        stream = anonymizer.anonymize_stream(texts=texts, **runtime_config)
        for pos, result in enumerate(stream):
            output_handler.output(result, idx=record_indices[pos], **metadata)
            processed += 1
    
    elapsed = time.time() - run_start
    return processed, elapsed


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch Anonymization")
    data_keys = add_data_args(parser)
    model_keys = add_model_args(parser)
    runtime_keys = add_runtime_args(parser)
    
    args = parser.parse_args()
    data_kwargs = {k: getattr(args, k) for k in data_keys}
    model_kwargs = {k: getattr(args, k) for k in model_keys}
    runtime_kwargs = {k: getattr(args, k) for k in runtime_keys}
    runtime_bundle = load_runtime_bundle(runtime_kwargs.pop("runtime_in", None))
    
    if args.list_annotations:
        print(f"Available annotations for {args.data}/{args.model}:")
        timestamps = list_batch_timestamps(dataset=args.data, model=args.model)
        for ts in (timestamps or ["(none)"]):
            if ts != "(none)":
                annotations = read_batch_annotations(args.data, args.model, ts)
                print(f"  - {ts} ({len(annotations)} records)")
        exit(0)
    
    records = load_data(data_kwargs)
    model_config = load_config(args.model_in)
    precompute_config = extract_precompute_config(model_config)
    capabilities = get_capabilities(args.model)
    explainer_config = extract_explainer_config(model_config)
    
    model_cls = MODEL_REGISTRY[model_kwargs.pop("model")]
    model = model_cls(**model_config, **model_kwargs, **data_kwargs)
    
    if capabilities.must_use_dataset:
        model.add_dataset_records(records)
    
    if capabilities.can_use_annotations and args.annotations_in:
        loaded_annotations = load_annotations(args.annotations_in, records)
        print(f"✓ Loaded annotations for {len(loaded_annotations)} records")
        if hasattr(model, 'set_annotations'):
            model.set_annotations(loaded_annotations, name=args.annotations)
    
    configure_model(model, model_config, explainer_config, runtime_bundle, capabilities, args.model, records)
    
    output_handler_cls = OUTPUT_HANDLER_REGISTRY.get(args.output, OUTPUT_HANDLER_REGISTRY["print"])
    batch_timestamp = args.timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    output_handler = output_handler_cls(timestamp=batch_timestamp) if args.output == "jsonl" else output_handler_cls()
    
    buckets = initialize_builder_params(model, runtime_bundle)
    
    dataset_indices_all = compute_dataset_indices(len(records), data_kwargs)
    index_lookup = {idx: pos for pos, idx in enumerate(dataset_indices_all)}
    
    texts_arg = runtime_kwargs.pop("texts", None)
    indices_arg = runtime_kwargs.pop("indices", None)
    
    if texts_arg and indices_arg:
        raise ValueError("Cannot specify both --texts and --indices")
    
    if capabilities.must_use_dataset:
        if texts_arg:
            raise ValueError(f"{args.model} requires dataset records, use --indices")
        dataset_indices = resolve_requested_indices(dataset_indices_all, indices_arg)
        record_indices = indices_arg or dataset_indices_all
    else:
        selected_indices = resolve_requested_indices(dataset_indices_all, indices_arg)
        if texts_arg:
            record_indices = list(range(len(texts_arg)))
            texts_or_indices = texts_arg
        else:
            texts_or_indices = [records[index_lookup[idx]].text for idx in selected_indices]
            record_indices = selected_indices
        dataset_indices = None
    
    stream_enabled = runtime_kwargs.pop("stream", False)
    metadata = {
        "dataset": args.data,
        "model": args.model,
        "task_id": args.start,
        "unique_name": args.unique_name
    }

    pre_inputs = dataset_indices if capabilities.must_use_dataset else texts_or_indices
    pre_risk_scores = None
    if precompute_config.get("risk_scores"):
        pre_risk_scores = load_precomputed_risk(precompute_config["risk_scores"])
    pre_start = time.time()
    try:
        model.pre_stream_anonymize(texts_or_indices=pre_inputs, risk_scores=pre_risk_scores)
    except NotImplementedError:
        pass
    pre_elapsed = time.time() - pre_start
    print(f"✓ Pre-computation before anonymization completed in {pre_elapsed:.2f}s")

    run_start = time.time()
    processed = 0
    
    if capabilities.must_use_dataset:
        for abs_idx, local_idx in zip(record_indices, dataset_indices):
            result = model.anonymize_from_dataset(idx=local_idx, buckets=buckets)
            output_handler.output(result, idx=abs_idx, **metadata)
            processed += 1
    else:
        for pos, text_or_idx in enumerate(texts_or_indices):
            result = model.anonymize(text_or_idx=text_or_idx, buckets=buckets)
            output_handler.output(result, idx=record_indices[pos], **metadata)
            processed += 1
    
    total_time = time.time() - run_start
    
    avg_time = total_time / processed if processed > 0 else 0
    print(f"\n{'='*80}\nAnonymization Performance:\n  Total time: {total_time:.2f}s\n  Texts processed: {processed}\n  Average time per text: {avg_time:.2f}s\n  Throughput: {processed/total_time:.2f} texts/s" if total_time > 0 and processed > 0 else "\n  Throughput: N/A")
    print('='*80)
    
    if hasattr(output_handler, 'close'):
        output_handler.close()
