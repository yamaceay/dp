from typing import Any, Dict, Optional, List, Tuple, Union, Callable
import argparse
import json
import yaml
import time
import itertools
from datetime import datetime

from dp.methods.anonymizer import Anonymizer, AnonymizationBuilder
from dp.methods.registry import MODEL_REGISTRY
from dp.methods.constants import get_capabilities, PII_CLASSIFIER_MODEL_LIST, RISK_MASKER_MODEL_LIST
from dp.loaders import ADAPTER_REGISTRY, DatasetRecord, read_batch_annotations, read_batch_annotations_from_path, list_batch_timestamps
from dp.utils.pii_detector import PIIDetector
from dp.utils.selector import PIIOnlySelector, AllSelector, ByRiskSelector
from dp.utils.explainer import UniformExplainer, GreedyExplainer, ShapExplainer
from dp.utils.output import OUTPUT_HANDLER_REGISTRY
from runtime import load_runtime_bundle

available_models = list(MODEL_REGISTRY.keys())
available_datasets = list(ADAPTER_REGISTRY.keys())
pii_confidence_models = set(PII_CLASSIFIER_MODEL_LIST)
risk_tolerance_models = set(RISK_MASKER_MODEL_LIST)


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
    adapter = ADAPTER_REGISTRY.get(data_kwargs["data"])
    if adapter is None:
        raise ValueError(f"Adapter '{data_kwargs['data']}' not found.")
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


def load_model(model_config: dict, model_kwargs: dict, data_kwargs: dict, records: List[DatasetRecord]) -> Anonymizer:
    model_name = model_kwargs["model"]
    model_cls = MODEL_REGISTRY.get(model_name)
    if model_cls is None:
        raise ValueError(f"Model '{model_name}' not found.")
    
    capabilities = get_capabilities(model_name)
    model_instance = model_cls(**model_config, **model_kwargs, **data_kwargs)
    
    if capabilities.must_use_dataset:
        model_instance.add_dataset_records(records)
    
    return model_instance


def compute_dataset_indices(dataset_len: int, data_kwargs: Dict[str, Any]) -> List[int]:
    start = data_kwargs.get("start") or 0
    end = data_kwargs.get("end") or dataset_len
    step = data_kwargs.get("step") or 1
    end = min(end, dataset_len)
    
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
        raise ValueError(f"Requested indices {missing} not in available dataset slice.")
    return requested


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


class StreamProcessor:
    def __init__(self, capabilities, anonymizer: Anonymizer, builder: AnonymizationBuilder, output_handler, runtime_config: dict, **metadata):
        self.capabilities = capabilities
        self.anonymizer = anonymizer
        self.builder = builder
        self.output_handler = output_handler
        self.runtime_config = runtime_config
        self.metadata = metadata
        self.processed = 0
    
    def process(self) -> Tuple[int, float]:
        run_start = time.time()
        
        if self.capabilities.requires_k:
            self._process_k_anonymity()
        elif self.capabilities.must_use_dataset:
            self._process_dataset_based()
        elif self.capabilities.requires_epsilon:
            self._process_epsilon()
        else:
            self._process_generic()
        
        elapsed = time.time() - run_start
        return self.processed, elapsed
    
    def _process_k_anonymity(self):
        stream = self.builder.anonymize_stream(**self.runtime_config)
        for idx_value, per_idx in zip(self.metadata["record_indices"], stream):
            for results in per_idx.values():
                for result in results:
                    self._output(result, idx_value)
            self.processed += 1
    
    def _process_dataset_based(self):
        for abs_idx, local_idx in zip(self.metadata["record_indices"], self.metadata["dataset_indices"]):
            result = self.anonymizer.anonymize_from_dataset(idx=local_idx, **self.runtime_config)
            self._output(result, abs_idx)
            self.processed += 1
    
    def _process_epsilon(self):
        stream = self.builder.anonymize_stream(**self.runtime_config)
        for idx_value, per_text in zip(self.metadata["record_indices"], stream):
            for results in per_text.values():
                for result in results:
                    self._output(result, idx_value)
            self.processed += 1
    
    def _process_generic(self):
        param_name = "pii_confidence" if self.capabilities.is_pii_classifier else ("risk_tolerance" if self.capabilities.is_risk_masker else None)
        values = self._get_parameter_values(param_name)
        setter = getattr(self.anonymizer, f"set_{param_name}", None) if param_name else None
        
        for value in values:
            if setter and value is not None:
                setter(value)
            
            stream = self.anonymizer.anonymize_stream(texts=self.metadata["texts"], **self.runtime_config)
            for pos, result in enumerate(stream):
                if param_name and value is not None:
                    result.metadata = {**dict(result.metadata or {}), param_name: value}
                self._output(result, self.metadata["record_indices"][pos])
                self.processed += 1
    
    def _get_parameter_values(self, param_name: Optional[str]) -> List[Optional[float]]:
        if not param_name:
            return [None]
        values = getattr(self.metadata.get("runtime_bundle"), f"{param_name}_values", None) or [None]
        return list(dict.fromkeys(values))
    
    def _output(self, result, idx):
        self.output_handler.output(result, idx=idx, **{k: v for k, v in self.metadata.items() if k not in ["record_indices", "dataset_indices", "texts", "runtime_bundle"]})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch Anonymization - Process Multiple Records")
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
    capabilities = get_capabilities(args.model)
    
    explainer_block = model_config.pop("explainer", {}) or {}
    model_config["risk_temperature"] = explainer_block.get("risk_temperature")
    
    model = load_model(model_config, model_kwargs, data_kwargs, records)
    
    if capabilities.can_use_annotations and args.annotations_in:
        loaded_annotations = {}
        for source in args.annotations_in.split(','):
            for idx, annotations in enumerate(read_batch_annotations_from_path(source.strip())):
                if annotations and idx < len(records):
                    uid = str(records[idx].uid if hasattr(records[idx], 'uid') else idx)
                    loaded_annotations.setdefault(uid, []).extend(annotations)
        print(f"✓ Loaded annotations for {len(loaded_annotations)} records")
        if hasattr(model, 'set_annotations'):
            model.set_annotations(loaded_annotations, name=args.annotations)
    
    if capabilities.can_use_filtering:
        selector_config = model_config.get("token_selection", {})
        selector_type = selector_config.get("name")
        
        if selector_type == "pii_only":
            pii_path = selector_config.get("pii_annotator")
            if not pii_path:
                raise ValueError("PIIOnlySelector requires 'pii_annotator' in config.")
            selector = PIIOnlySelector(PIIDetector(model_name=pii_path, use_chunking=selector_config.get("pii_chunking", {}).get("enabled", False)), threshold=selector_config.get("pii_threshold"))
        elif selector_type == "by_risk":
            if "risk_tolerance" not in selector_config:
                raise ValueError("ByRiskSelector requires 'risk_tolerance'.")
            selector = ByRiskSelector(risk_tolerance=selector_config["risk_tolerance"])
        else:
            selector = AllSelector()
        
        model.set_filtering_strategy(selector)
    
    if capabilities.can_use_scoring:
        explainer_name = explainer_block.get("name", "uniform")
        
        if capabilities.must_use_non_uniform_explainer and explainer_name == "uniform":
            raise ValueError(f"{args.model} requires non-uniform explainability.")
        
        if explainer_name == "uniform":
            model.set_scoring_strategy(UniformExplainer())
        else:
            explainer_path = explainer_block.get("tri_pipeline")
            if not explainer_path:
                raise ValueError(f"{args.model} requires tri_pipeline path.")
            
            tri_chunking_enabled = model_config.get("tri_chunking", {}).get("enabled", False)
            if explainer_name == "greedy":
                model.set_scoring_strategy(GreedyExplainer(model_name=explainer_path, use_chunking=tri_chunking_enabled))
            elif explainer_name == "shap":
                model.set_scoring_strategy(ShapExplainer(model_name=explainer_path, use_chunking=tri_chunking_enabled))
            else:
                raise ValueError(f"Unknown explainability: {explainer_name}")
    
    risk_path = explainer_block.get("risk_scores") or runtime_bundle.base_config.get("explainer", {}).get("risk_scores")
    if risk_path and hasattr(model, "set_risk_scores"):
        model.set_risk_scores(load_precomputed_risk(risk_path), records=records)
    
    output_handler_cls = OUTPUT_HANDLER_REGISTRY.get(args.output, OUTPUT_HANDLER_REGISTRY["print"])
    batch_timestamp = args.timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    output_handler = output_handler_cls(timestamp=batch_timestamp) if args.output == "jsonl" else output_handler_cls()
    
    dataset_indices_all = compute_dataset_indices(len(records), data_kwargs)
    index_lookup = {idx: pos for pos, idx in enumerate(dataset_indices_all)}
    builder = model.builder()
    
    texts_arg = runtime_kwargs.pop("texts", None)
    indices_arg = runtime_kwargs.pop("indices", None)
    
    if texts_arg and indices_arg:
        raise ValueError("Cannot specify both --texts and --indices")
    
    if capabilities.must_use_dataset:
        if texts_arg:
            raise ValueError(f"{args.model} requires dataset records, use --indices")
        dataset_indices = resolve_requested_indices(dataset_indices_all, indices_arg)
        builder.with_indices(dataset_indices)
        record_indices = indices_arg or dataset_indices_all
    else:
        selected_indices = resolve_requested_indices(dataset_indices_all, indices_arg)
        if texts_arg:
            texts = texts_arg
            record_indices = list(range(len(texts)))
        else:
            texts = [records[index_lookup[idx]].text for idx in selected_indices]
            record_indices = selected_indices
        builder.with_texts(texts)
        dataset_indices = None
    
    if args.model in pii_confidence_models and runtime_bundle.pii_confidence_values:
        builder.with_pii_confidences(runtime_bundle.pii_confidence_values)
    
    if args.model in risk_tolerance_models and runtime_bundle.risk_tolerance_values:
        builder.with_risk_tolerances(runtime_bundle.risk_tolerance_values)
    
    if capabilities.requires_k:
        builder.with_ks(runtime_bundle.k_values or [5])
    
    if capabilities.requires_epsilon:
        builder.with_epsilons(runtime_bundle.epsilon_values or [100.0])
    
    stream_enabled = runtime_kwargs.pop("stream", False)
    if stream_enabled:
        if args.output != "jsonl":
            raise ValueError("--stream only works with --output jsonl")
        if not capabilities.supports_streaming:
            raise ValueError(f"{args.model} does not support streaming")
        
        processor = StreamProcessor(
            capabilities, model, builder, output_handler, runtime_bundle.base_config,
            dataset=args.data, model=args.model, task_id=args.start, unique_name=args.unique_name,
            record_indices=record_indices, dataset_indices=dataset_indices, texts=texts if not capabilities.must_use_dataset else None,
            runtime_bundle=runtime_bundle
        )
        processed, total_time = processor.process()
    else:
        start_time = time.time()
        results = [r for r in builder.anonymize(**runtime_bundle.base_config)]
        total_time = time.time() - start_time
        
        flat_res, flat_indices = flatten_results(results, record_indices)
        output_results(flat_res, flat_indices, output_handler, verbose=args.output != "jsonl", 
                      dataset=args.data, model=args.model, task_id=args.start, unique_name=args.unique_name)
        processed = len(record_indices)
    
    avg_time = total_time / processed if processed > 0 else 0
    print(f"\n{'='*80}\nAnonymization Performance:\n  Total time: {total_time:.2f}s\n  Texts processed: {processed}\n  Average time per text: {avg_time:.2f}s\n  Throughput: {processed/total_time:.2f} texts/s" if total_time > 0 and processed > 0 else "\n  Throughput: N/A")
    print('='*80)
    
    if hasattr(output_handler, 'close'):
        output_handler.close()
