from typing import Any, Dict, Optional, List, Tuple, Union
import argparse
import json
import yaml
import time
from datetime import datetime
import fnmatch
from pathlib import Path

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
    parser.add_argument('--unique_name', type=str, default=None)
    parser.add_argument('--as_task', action='store_true')
    return ['runtime_in', 'texts', 'indices', 'output', 'annotations_in', 'list_annotations', 'unique_name', 'as_task']


def load_config(path: Optional[str]) -> dict:
    if path is None:
        return {}
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f) or {}


def _normalize_starting_anonymizations(value: object) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        paths: List[str] = []
        for item in value:
            if not isinstance(item, str):
                raise ValueError("starting_anonymizations entries must be strings")
            paths.append(item)
        return paths
    raise ValueError("starting_anonymizations must be a string or list of strings")


def extract_starting_anonymizations_config(model_config: dict) -> List[str]:
    value = model_config.pop("starting_anonymizations", None)
    if value is None:
        value = model_config.pop("starting_anonymization", None)
    return _normalize_starting_anonymizations(value)


def _normalize_param_patterns(value: object) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        patterns: List[str] = []
        for item in value:
            if not isinstance(item, str):
                raise ValueError("params entries must be strings")
            patterns.append(item)
        return patterns
    raise ValueError("params entries must be a string or list of strings")


def _match_any(value: str, patterns: List[str]) -> bool:
    if not patterns:
        return False
    for pat in patterns:
        if fnmatch.fnmatch(value, pat):
            return True
    return False


def _runtime_suffix_from_source(path: str) -> Optional[Tuple[str, str]]:
    p = Path(path)
    name = p.name
    if name.startswith("eps_") and name.endswith(".yaml") and p.parent.name == "dp":
        return "epsilon", name[len("eps_") : -len(".yaml")]
    if name.startswith("lambda_") and name.endswith(".yaml") and p.parent.name == "pii_confidence":
        return "lambdas", name[len("lambda_") : -len(".yaml")]
    if name.startswith("rho_") and name.endswith(".yaml") and p.parent.name == "risk_tolerance":
        return "rhos", name[len("rho_") : -len(".yaml")]
    if name.startswith("k_") and name.endswith(".yaml") and p.parent.name == "k_anon":
        return "ks", name[len("k_") : -len(".yaml")]
    return None


def validate_runtime_params(model_config: dict, runtime_bundle: object) -> None:
    params_obj = model_config.get("params")
    if params_obj is None:
        return
    if not isinstance(params_obj, dict):
        raise ValueError("params must be a mapping")

    allowed: Dict[str, List[str]] = {
        "epsilon": _normalize_param_patterns(params_obj.get("epsilon")),
        "lambdas": _normalize_param_patterns(params_obj.get("lambdas")),
        "rhos": _normalize_param_patterns(params_obj.get("rhos")),
        "ks": _normalize_param_patterns(params_obj.get("ks")),
    }

    sources = getattr(runtime_bundle, "sources", None)
    if sources is None:
        return
    if not isinstance(sources, list):
        raise ValueError("runtime sources must be a list")

    eps_seen = 0
    for src in sources:
        if not isinstance(src, str):
            continue
        parsed = _runtime_suffix_from_source(src)
        if parsed is None:
            continue
        key, suffix = parsed
        patterns = allowed.get(key, [])
        if not patterns:
            raise ValueError(f"Runtime config '{src}' sets '{key}', which is not allowed by model params")
        if not _match_any(suffix, patterns):
            raise ValueError(f"Runtime config '{src}' value '{suffix}' is not allowed for '{key}'")
        if key == "epsilon":
            eps_seen += 1
            if eps_seen > 1:
                raise ValueError("epsilon must be provided as a single runtime param")


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


def build_selector(selector_config: dict, runtime_bundle=None):
    selector_type = selector_config.get("name")
    
    if selector_type == "pii_only":
        pii_path = selector_config.get("pii_annotator")
        if not pii_path:
            raise ValueError("PIIOnlyUnit requires 'pii_annotator' in config")
        pii_chunking = selector_config.get("pii_chunking", {}).get("enabled", False)
        detector = PIIDetector(model_name=pii_path, use_chunking=pii_chunking)
        unit = PIIOnlySelector(pii_detector=detector)
        if runtime_bundle and hasattr(runtime_bundle, 'pii_confidence_values') and runtime_bundle.pii_confidence_values:
            unit.set_thresholds(runtime_bundle.pii_confidence_values, name="lambda")
        return unit
    
    if selector_type == "by_risk":
        temperature = selector_config.get("risk_temperature", 1.0)
        unit = ByRiskSelector(temperature=temperature)
        if runtime_bundle and hasattr(runtime_bundle, 'risk_tolerance_values') and runtime_bundle.risk_tolerance_values:
            unit.set_thresholds(runtime_bundle.risk_tolerance_values, name="rho")
        return unit
    
    if selector_type == "until_k":
        unit = UntilKSelector()
        if runtime_bundle and hasattr(runtime_bundle, 'k_values') and runtime_bundle.k_values:
            unit.set_thresholds(runtime_bundle.k_values, name="k")
        return unit
    
    return AllSelector()


def build_explainer(explainer_config: dict, model_config: dict, capabilities, model_name: str):
    explainer_name = explainer_config["name"]
    
    if capabilities.must_use_scoring and explainer_name == "uniform":
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
    selector_config = extract_selector_config(model_config)
    if explainer_config.get("risk_temperature") is not None:
        if isinstance(selector_config, dict) and selector_config.get("name") == "by_risk":
            selector_config = {"risk_temperature": explainer_config["risk_temperature"], **selector_config}
    
    if capabilities.must_use_pii_selector:
        if not selector_config or selector_config.get("name") != "pii_only":
            raise ValueError(f"{model_name} requires pii_only selector")
        model.set_filtering_strategy(build_selector(selector_config, runtime_bundle))
    elif capabilities.must_use_risk_selector:
        if not selector_config or selector_config.get("name") != "by_risk":
            raise ValueError(f"{model_name} requires by_risk selector")
        model.set_filtering_strategy(build_selector(selector_config, runtime_bundle))
    elif capabilities.must_use_k_selector:
        if not selector_config or selector_config.get("name") != "until_k":
            raise ValueError(f"{model_name} requires until_k selector")
        model.set_filtering_strategy(build_selector(selector_config, runtime_bundle))
    elif capabilities.can_use_pii_selector or capabilities.can_use_risk_selector or capabilities.can_use_k_selector:
        if selector_config:
            model.set_filtering_strategy(build_selector(selector_config, runtime_bundle))
    
    if capabilities.must_use_scoring or capabilities.can_use_scoring:
        model.set_scoring_strategy(build_explainer(explainer_config, model_config, capabilities, model_name))


def load_annotations(annotations_in: str, records: List[DatasetRecord]) -> Dict[str, List]:
    loaded_annotations = {}
    for source in annotations_in.split(','):
        for idx, annotations in enumerate(read_batch_annotations_from_path(source.strip())):
            if annotations and idx < len(records):
                uid = str(records[idx].uid if hasattr(records[idx], 'uid') else idx)
                loaded_annotations.setdefault(uid, []).extend(annotations)
    return loaded_annotations


def load_starting_anonymizations(paths: List[str], records: List[DatasetRecord]) -> Dict[str, List[dict]]:
    annotations: Dict[str, List[dict]] = {}
    for path in paths:
        with open(path, "r", encoding="utf-8") as reader:
            for line_num, raw in enumerate(reader, start=1):
                entry = raw.strip()
                if not entry:
                    continue
                payload = json.loads(entry)
                spans = payload.get("spans") or []
                if not isinstance(spans, list):
                    raise ValueError(f"Invalid spans in '{path}' at line {line_num}")

                uid = payload.get("uid")
                if uid is None and isinstance(payload.get("metadata"), dict):
                    uid = payload["metadata"].get("uid")

                if uid is None:
                    idx = payload.get("idx")
                    if idx is None:
                        raise ValueError(f"Missing 'uid'/'metadata.uid' and 'idx' in '{path}' at line {line_num}")
                    if not isinstance(idx, int):
                        raise ValueError(f"Invalid 'idx' in '{path}' at line {line_num}")
                    if idx < 0 or idx >= len(records):
                        raise ValueError(
                            f"Index {idx} in '{path}' at line {line_num} is out of bounds for dataset (len={len(records)})"
                        )
                    rec = records[idx]
                    uid = str(getattr(rec, "uid", idx))
                else:
                    uid = str(uid)

                for span in spans:
                    if not isinstance(span, dict):
                        continue
                    if "start" not in span or "end" not in span:
                        continue
                    annotations.setdefault(uid, []).append({"start": int(span["start"]), "end": int(span["end"])})
    return annotations


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
    from dp.methods.constants import KParams, LambdaParams, RhoParams
    from dp.methods.constants import EpsilonParam
    model_name = getattr(anonymizer, "MODEL_NAME", None)

    if model_name == "dpmlm":
        if getattr(runtime_bundle, "epsilon_value", None) is None:
            return []
        return [EpsilonParam(epsilon=runtime_bundle.epsilon_value)]

    if model_name == "baroud":
        if not getattr(runtime_bundle, "pii_confidence_values", None):
            return []
        return [LambdaParams(lambdas=runtime_bundle.pii_confidence_values)]

    if model_name == "risk":
        if not getattr(runtime_bundle, "risk_tolerance_values", None):
            return []
        return [RhoParams(rhos=runtime_bundle.risk_tolerance_values)]

    if model_name == "petre":
        if not getattr(runtime_bundle, "k_values", None):
            return []
        return [KParams(ks=runtime_bundle.k_values)]

    if getattr(runtime_bundle, "epsilon_value", None) is not None:
        return [EpsilonParam(epsilon=runtime_bundle.epsilon_value)]
    if getattr(runtime_bundle, "k_values", None):
        return [KParams(ks=runtime_bundle.k_values)]
    if getattr(runtime_bundle, "pii_confidence_values", None):
        return [LambdaParams(lambdas=runtime_bundle.pii_confidence_values)]
    if getattr(runtime_bundle, "risk_tolerance_values", None):
        return [RhoParams(rhos=runtime_bundle.risk_tolerance_values)]
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
    starting_anonymization_paths = extract_starting_anonymizations_config(model_config)
    validate_runtime_params(model_config, runtime_bundle)
    precompute_config = extract_precompute_config(model_config)
    capabilities = get_capabilities(args.model)
    explainer_config = extract_explainer_config(model_config)

    selector_config = extract_selector_config(model_config)
    dpmlm_selector_name = selector_config.get("name") if isinstance(selector_config, dict) else None
    dpmlm_has_precomputed_risk = bool(precompute_config.get("risk_scores"))
    dpmlm_has_tri_explainer = bool(explainer_config.get("tri_pipeline"))
    dpmlm_requires_dataset = (
        args.model == "dpmlm"
        and (
            dpmlm_selector_name == "until_k"
            or dpmlm_has_precomputed_risk
            or dpmlm_has_tri_explainer
        )
    )
    
    model_cls = MODEL_REGISTRY[model_kwargs.pop("model")]
    model = model_cls(**model_config, **model_kwargs, **data_kwargs)
    
    if capabilities.must_use_dataset or dpmlm_requires_dataset:
        if not hasattr(model, "add_dataset_records"):
            raise ValueError(f"{args.model} requires dataset records for this configuration")
        model.add_dataset_records(records)

    merged_annotations: Optional[Dict[str, List]] = None
    if starting_anonymization_paths:
        if args.texts:
            raise ValueError("starting_anonymizations cannot be used with --texts")
        if not hasattr(model, "set_annotations"):
            raise ValueError(f"{args.model} does not support starting_anonymizations")
        starting_annotations = load_starting_anonymizations(starting_anonymization_paths, records)
        print(f"✓ Loaded starting anonymizations for {len(starting_annotations)} records")
        merged_annotations = dict(starting_annotations)
    
    if capabilities.can_use_annotations and args.annotations_in:
        loaded_annotations = load_annotations(args.annotations_in, records)
        print(f"✓ Loaded annotations for {len(loaded_annotations)} records")
        if hasattr(model, 'set_annotations'):
            if merged_annotations is None:
                merged_annotations = dict(loaded_annotations)
            else:
                for uid, items in loaded_annotations.items():
                    merged_annotations.setdefault(uid, []).extend(items)

    if merged_annotations is not None and hasattr(model, 'set_annotations'):
        model.set_annotations(merged_annotations, name=args.annotations or ("starting" if starting_anonymization_paths else None))
    
    configure_model(model, model_config, explainer_config, runtime_bundle, capabilities, args.model, records)
    
    output_handler_cls = OUTPUT_HANDLER_REGISTRY.get(args.output, OUTPUT_HANDLER_REGISTRY["print"])
    batch_timestamp = args.timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    output_handler = output_handler_cls(timestamp=batch_timestamp) if args.output == "jsonl" else output_handler_cls()
    
    buckets = initialize_builder_params(model, runtime_bundle)

    runtime_args: Dict[str, Any] = {}
    if getattr(runtime_bundle, 'epsilon_value', None) is not None:
        runtime_args['epsilon'] = runtime_bundle.epsilon_value
    if getattr(runtime_bundle, 'k_values', None):
        runtime_args['ks'] = runtime_bundle.k_values
    if getattr(runtime_bundle, 'pii_confidence_values', None):
        runtime_args['lambdas'] = runtime_bundle.pii_confidence_values
    if getattr(runtime_bundle, 'risk_tolerance_values', None):
        runtime_args['rhos'] = runtime_bundle.risk_tolerance_values
    
    dataset_indices_all = compute_dataset_indices(len(records), data_kwargs)
    index_lookup = {idx: pos for pos, idx in enumerate(dataset_indices_all)}
    
    texts_arg = runtime_kwargs.pop("texts", None)
    indices_arg = runtime_kwargs.pop("indices", None)
    
    if texts_arg and indices_arg:
        raise ValueError("Cannot specify both --texts and --indices")
    
    record_names_for_precompute: Optional[List[str]] = None
    if capabilities.must_use_dataset or dpmlm_requires_dataset:
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
            selected_records = [records[index_lookup[idx]] for idx in selected_indices]
            texts_or_indices = [r.text for r in selected_records]
            record_names_for_precompute = [str(r.uid) for r in selected_records]
            record_indices = selected_indices
        dataset_indices = None
    
    metadata = {
        "dataset": args.data,
        "model": args.model,
        "unique_name": args.unique_name
    }
    if args.as_task:
        metadata["task_id"] = args.start

    anonymization_inputs = dataset_indices if (capabilities.must_use_dataset or dpmlm_requires_dataset) else texts_or_indices
    pre_risk_scores = None
    if precompute_config.get("risk_scores"):
        pre_risk_scores = load_precomputed_risk(precompute_config["risk_scores"])
    pre_start = time.time()
    pre_kwargs: Dict[str, Any] = {}
    if record_names_for_precompute is not None:
        pre_kwargs["record_names"] = record_names_for_precompute
    model.pre_stream_anonymize(texts_or_indices=anonymization_inputs, risk_scores=pre_risk_scores, **pre_kwargs)
    pre_elapsed = time.time() - pre_start
    print(f"✓ Pre-computation before anonymization completed in {pre_elapsed:.2f}s")

    run_start = time.time()
    processed = 0

    stream = model.stream_anonymize(texts_or_indices=anonymization_inputs, buckets=buckets)
    for abs_idx, result_list in zip(record_indices, stream):
        for hp, result in result_list:
            output_handler.output(result, idx=abs_idx, **metadata, hyperparams=hp)
        processed += 1

    total_time = time.time() - run_start
    
    avg_time = total_time / processed if processed > 0 else 0
    print(f"\n{'='*80}\nAnonymization Performance:\n  Total time: {total_time:.2f}s\n  Texts processed: {processed}\n  Average time per text: {avg_time:.2f}s\n  Throughput: {processed/total_time:.2f} texts/s" if total_time > 0 and processed > 0 else "\n  Throughput: N/A")
    print('='*80)
    
    if hasattr(output_handler, 'close'):
        output_handler.close()
