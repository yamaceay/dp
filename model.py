from typing import Any, Dict, Optional, List, Tuple, Union
import argparse
import json
import yaml
import time
from datetime import datetime
import fnmatch
from pathlib import Path

from dp.methods.anonymizer import Anonymizer
from dp.methods.registry import MODEL_REGISTRY, get_capabilities
from dp.loaders import ADAPTER_REGISTRY, DatasetRecord, get_adapter
from dp.loaders.results import build_dataset_from_results
from dp.utils.pii_detector import PIIDetector
from dp.utils.selector import PIIOnlyUnit, AllUnit, ByRiskUnit, UntilKUnit
from dp.utils.explainer import UniformExplainer, ShapExplainer, ShapType
from dp.utils.output import OUTPUT_HANDLER_REGISTRY
from runtime import load_runtime_bundle
from dp.utils.tasking import resolve_task_id, apply_task_template

available_models = list(MODEL_REGISTRY.keys())
available_datasets = list(ADAPTER_REGISTRY.keys())
SPLIT_ENABLED_DATASETS = {"tab", "db_bio"}


def add_data_args(parser: argparse.ArgumentParser) -> List[str]:
    parser.add_argument('--data', type=str, default=None)
    parser.add_argument('--data_in', type=str, default=None)
    parser.add_argument('--result_in', type=str, default=None)
    parser.add_argument('--start', type=int, default=None)
    parser.add_argument('--end', type=int, default=None)
    parser.add_argument('--step', type=int, default=None)
    parser.add_argument('--max_records', type=int, default=None)
    parser.add_argument('--split', type=str, default=None)
    return ['data', 'data_in', 'result_in', 'start', 'end', 'step', 'max_records', 'split']


def add_model_args(parser: argparse.ArgumentParser) -> List[str]:
    parser.add_argument('--model', type=str, required=True, choices=available_models)
    parser.add_argument('--model_in', type=str, default=None)
    return ['model', 'model_in']


def add_runtime_args(parser: argparse.ArgumentParser) -> List[str]:
    parser.add_argument('--runtime_in', type=str, nargs='+', default=None)
    parser.add_argument('--texts', type=str, nargs='+')
    parser.add_argument('--indices', type=int, nargs='+')
    parser.add_argument('--output', type=str, default='print', choices=list(OUTPUT_HANDLER_REGISTRY.keys()))
    parser.add_argument('--base_output_path', type=str, default="outputs")
    parser.add_argument('--timestamp', type=str, default=None)
    parser.add_argument('--unique_name', type=str, default=None)
    parser.add_argument('--as_task', action='store_true')
    return ['runtime_in', 'texts', 'indices', 'output', 'base_output_path', 'timestamp', 'unique_name', 'as_task']


def load_config(path: Optional[str]) -> dict:
    if path is None:
        return {}
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f) or {}


def _parse_scalar(raw: str) -> Any:
    value = raw.strip()
    lowered = value.lower()
    if lowered in {"null", "none"}:
        return None
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    try:
        if value.startswith("0") and len(value) > 1 and value[1].isdigit() and not value.startswith("0."):
            raise ValueError
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def _apply_set_overrides(cfg: Dict[str, Any], overrides: Optional[List[str]]) -> None:
    if not overrides:
        return
    for spec in overrides:
        if not isinstance(spec, str) or "=" not in spec:
            raise ValueError(f"Invalid --set override (expected key=value): {spec!r}")
        key_raw, value_raw = spec.split("=", 1)
        key_raw = key_raw.strip()
        if not key_raw:
            raise ValueError(f"Invalid --set override (empty key): {spec!r}")
        value = _parse_scalar(value_raw)
        parts = [p.strip() for p in key_raw.split(".") if p.strip()]
        if not parts:
            raise ValueError(f"Invalid --set override (empty key path): {spec!r}")
        cur: Any = cfg
        for part in parts[:-1]:
            if not isinstance(cur, dict):
                raise ValueError(f"Invalid --set path (not a mapping at '{part}'): {key_raw}")
            nxt = cur.get(part)
            if nxt is None:
                nxt = {}
                cur[part] = nxt
            if not isinstance(nxt, dict):
                raise ValueError(f"Invalid --set path (existing non-mapping at '{part}'): {key_raw}")
            cur = nxt
        if not isinstance(cur, dict):
            raise ValueError(f"Invalid --set path (not a mapping): {key_raw}")
        cur[parts[-1]] = value


def _template_strings(value: Any, task_id: Optional[int]) -> Any:
    if isinstance(value, str):
        return apply_task_template(value, task_id)
    if isinstance(value, list):
        return [_template_strings(item, task_id) for item in value]
    if isinstance(value, dict):
        return {k: _template_strings(v, task_id) for k, v in value.items()}
    return value


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


def _data_source_for_model(model_name: str, has_result_in: bool) -> str:
    supports_anonymized = {"dpmlm", "petre", "risk"}
    if model_name in supports_anonymized and has_result_in:
        return "anonymized"
    return "original"


def load_data(data_kwargs: dict, model_name: str) -> Tuple[List[DatasetRecord], Optional[List[int]]]:
    data = data_kwargs.get("data")
    data_in = data_kwargs.get("data_in")
    result_in = data_kwargs.get("result_in")
    split = data_kwargs.get("split")
    data_source = _data_source_for_model(model_name, bool(result_in))
    if data_source == "original":
        if not data or not data_in:
            raise ValueError("data and data_in are required for original dataset loading")
        adapter = get_adapter(data, data=data, data_in=data_in, split=split)
        return list(adapter.iter_records()), None
    if not result_in:
        raise ValueError("result_in is required for anonymized dataset loading")
    if not data or not data_in:
        raise ValueError("data and data_in are required to align anonymized results with dataset records")
    original_records = list(get_adapter(data, data=data, data_in=data_in, split=split).iter_records())
    records, source_indices = build_dataset_from_results(result_in, original_records)
    return records, source_indices


def load_precomputed_risk(path: str) -> Dict[str, Dict[str, object]]:
    """Load precomputed risk scores from JSONL.
    
    Each line: {"uid": ..., "offsets": [...], "scores": [...]}
    Returns: {uid: {"offsets": [...], "scores": [...]}}
    """
    if not path:
        raise ValueError("path cannot be empty")
    result: Dict[str, Dict[str, object]] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            entry = line.strip()
            if not entry:
                continue
            payload = json.loads(entry)
            uid = payload.get("uid")
            if uid is None:
                raise ValueError("Missing uid in risk file")
            result[str(uid)] = {
                "offsets": payload.get("offsets", []),
                "scores": payload.get("scores", []),
            }
    return result


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
        "result_in": block.get("result_in"),
    }


def extract_selector_config(model_config: dict) -> Dict[str, Any]:
    return model_config.get("token_selection", {})


def build_selector(selector_config: dict, runtime_bundle=None):
    selector_type = selector_config.get("name")
    
    if selector_type == "pii_only":
        pii_path = selector_config.get("pii_annotator")
        if not pii_path:
            raise ValueError("PIIOnlyUnit requires 'pii_annotator' in config")
        pii_chunking = selector_config.get("pii_chunking", {}).get("enabled", False)
        detector = PIIDetector(model_name=pii_path, use_chunking=pii_chunking)
        unit = PIIOnlyUnit(pii_detector=detector)
        if runtime_bundle and hasattr(runtime_bundle, 'pii_confidence_values') and runtime_bundle.pii_confidence_values:
            unit.set_thresholds(runtime_bundle.pii_confidence_values, name="lambda")
        return unit
    
    if selector_type == "by_risk":
        temperature = selector_config.get("risk_temperature", 1.0)
        unit = ByRiskUnit(temperature=temperature)
        if runtime_bundle and hasattr(runtime_bundle, 'risk_tolerance_values') and runtime_bundle.risk_tolerance_values:
            unit.set_thresholds(runtime_bundle.risk_tolerance_values, name="rho")
        return unit
    
    if selector_type == "until_k":
        unit = UntilKUnit()
        if runtime_bundle and hasattr(runtime_bundle, 'k_values') and runtime_bundle.k_values:
            unit.set_thresholds(runtime_bundle.k_values, name="k")
        return unit
    
    return AllUnit()


def build_explainer(explainer_config: dict, model_config: dict, capabilities, model_name: str):
    explainer_name = explainer_config["name"]
    
    if capabilities.must_use_scoring and explainer_name == "uniform":
        raise ValueError(f"{model_name} requires non-uniform explainability")
    
    if explainer_name == "uniform":
        return UniformExplainer()
    
    tri_pipeline = explainer_config.get("tri_pipeline")
    if not tri_pipeline:
        raise ValueError(f"{model_name} requires tri_pipeline path")

    explainer_type = ShapType(explainer_name)

    return ShapExplainer(model_name=tri_pipeline, explainer_type=explainer_type)

def configure_model(model: Anonymizer, model_config: dict, explainer_config: dict, selector_config: dict, runtime_bundle, capabilities, model_name: str, records: List[DatasetRecord]):
    if explainer_config.get("risk_temperature") is not None:
        if "risk_temperature" not in selector_config:
            selector_config["temperature"] = explainer_config["risk_temperature"]
        if hasattr(model, "risk_temperature"):
            model.risk_temperature = float(explainer_config["risk_temperature"])
    
    if any([
        capabilities.must_use_pii_selector,
        capabilities.must_use_risk_selector,
        capabilities.must_use_k_selector
    ]) and not selector_config:
        raise ValueError(f"{model_name} requires a token selection strategy")
        
    if capabilities.must_use_pii_selector:
        if selector_config.get("name") != "pii_only":
            raise ValueError(f"{model_name} requires pii_only selector")
        
    elif capabilities.must_use_risk_selector:
        if selector_config.get("name") != "by_risk":
            raise ValueError(f"{model_name} requires by_risk selector")
    
    elif capabilities.must_use_k_selector:
        if selector_config.get("name") != "until_k":
            raise ValueError(f"{model_name} requires until_k selector")
    
    if selector_config:
        model.set_unit(build_selector(selector_config, runtime_bundle))

    if capabilities.must_use_scoring or capabilities.can_use_scoring:
        model.set_explainer(build_explainer(explainer_config, model_config, capabilities, model_name))

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


def map_output_indices(record_positions: List[int], source_indices: Optional[List[int]]) -> List[Optional[int]]:
    if source_indices is None:
        return [int(pos) for pos in record_positions]
    mapped: List[Optional[int]] = []
    for pos in record_positions:
        if pos < 0 or pos >= len(source_indices):
            raise ValueError(f"Record position {pos} is out of range for source indices")
        mapped.append(source_indices[pos])
    return mapped


def initialize_builder_params(anonymizer: Anonymizer, runtime_bundle):
    from dp.methods.constants import KParams, LambdaParams, RhoParams
    from dp.methods.constants import EpsilonParam
    from dp.utils.selector import ByRiskUnit, PIIOnlyUnit, UntilKUnit
    model_name = getattr(anonymizer, "MODEL_NAME", None)

    if model_name == "dpmlm":
        if getattr(runtime_bundle, "epsilon_value", None) is None:
            return []
        buckets = [EpsilonParam(epsilon=runtime_bundle.epsilon_value)]
        unit = getattr(anonymizer, "_unit", None)
        if isinstance(unit, UntilKUnit):
            if not getattr(runtime_bundle, "k_values", None):
                return buckets
            buckets.append(KParams(ks=runtime_bundle.k_values))
        elif isinstance(unit, ByRiskUnit):
            if not getattr(runtime_bundle, "risk_tolerance_values", None):
                return buckets
            buckets.append(RhoParams(rhos=runtime_bundle.risk_tolerance_values))
        elif isinstance(unit, PIIOnlyUnit):
            if not getattr(runtime_bundle, "pii_confidence_values", None):
                return buckets
            buckets.append(LambdaParams(lambdas=runtime_bundle.pii_confidence_values))
        return buckets

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
    parser.add_argument(
        "--set",
        dest="set_overrides",
        action="append",
        default=None,
    )
    parser.add_argument("--task_id", type=int, default=None)
    
    args = parser.parse_args()
    data_kwargs = {k: getattr(args, k) for k in data_keys}
    model_kwargs = {k: getattr(args, k) for k in model_keys}
    runtime_kwargs = {k: getattr(args, k) for k in runtime_keys}
    task_id = resolve_task_id(args.task_id)
    data_kwargs = _template_strings(data_kwargs, task_id)
    model_kwargs = _template_strings(model_kwargs, task_id)
    runtime_kwargs = _template_strings(runtime_kwargs, task_id)
    runtime_bundle = load_runtime_bundle(runtime_kwargs.pop("runtime_in", None))
    
    model_config = load_config(args.model_in)
    _apply_set_overrides(model_config, args.set_overrides)
    model_config = _template_strings(model_config, task_id)
    precompute_config = extract_precompute_config(model_config)
    if data_kwargs.get("result_in") is None and precompute_config.get("result_in") is not None:
        data_kwargs["result_in"] = precompute_config.get("result_in")
    records, source_indices = load_data(data_kwargs, args.model)
    validate_runtime_params(model_config, runtime_bundle)
    capabilities = get_capabilities(args.model)
    explainer_config = extract_explainer_config(model_config)

    selector_config = extract_selector_config(model_config)
    dpmlm_selector_name = selector_config.get("name") if isinstance(selector_config, dict) else None
    dpmlm_has_precomputed_risk = bool(precompute_config.get("risk_scores"))
    dpmlm_has_tri_explainer = bool(explainer_config.get("tri_pipeline"))
    dpmlm_has_result_in = bool(data_kwargs.get("result_in"))
    dpmlm_requires_dataset = (
        args.model == "dpmlm"
        and (
            dpmlm_selector_name == "until_k"
            or dpmlm_has_precomputed_risk
            or dpmlm_has_tri_explainer
            or dpmlm_has_result_in
        )
    )
    
    model_cls = MODEL_REGISTRY[model_kwargs.pop("model")]
    model = model_cls(**model_config, **model_kwargs, **data_kwargs)
    risk_has_result_in = args.model == "risk" and bool(data_kwargs.get("result_in"))
    
    if capabilities.must_use_dataset or dpmlm_requires_dataset or risk_has_result_in:
        if not hasattr(model, "add_dataset_records"):
            raise ValueError(f"{args.model} requires dataset records for this configuration")
        model.add_dataset_records(records)

    configure_model(model, model_config, explainer_config, selector_config, runtime_bundle, capabilities, args.model, records)
    
    output_handler_cls = OUTPUT_HANDLER_REGISTRY.get(args.output, OUTPUT_HANDLER_REGISTRY["print"])
    batch_timestamp = args.timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    output_handler = output_handler_cls(timestamp=batch_timestamp, base_path=args.base_output_path) if args.output == "jsonl" else output_handler_cls()
    
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
    split_arg = data_kwargs.get("split")
    split_dataset = str(data_kwargs.get("data") or "").lower()
    split_is_active = bool(split_arg) and split_dataset in SPLIT_ENABLED_DATASETS
    if split_arg and split_dataset not in SPLIT_ENABLED_DATASETS:
        print(f"split is ignored for dataset '{split_dataset}'")
    if split_is_active:
        texts_arg = None
        indices_arg = None
        print("split is set; ignoring --texts and --indices")
    
    if texts_arg and indices_arg:
        raise ValueError("Cannot specify both --texts and --indices")
    
    record_names_for_precompute: Optional[List[str]] = None
    prior_edits_for_precompute: Optional[List[List[Dict[str, object]]]] = None
    record_positions: List[int]
    if capabilities.must_use_dataset or dpmlm_requires_dataset:
        if texts_arg:
            raise ValueError(f"{args.model} requires dataset records, use --indices")
        dataset_indices = resolve_requested_indices(dataset_indices_all, indices_arg)
        record_positions = indices_arg or dataset_indices_all
    else:
        selected_indices = resolve_requested_indices(dataset_indices_all, indices_arg)
        if texts_arg:
            record_positions = list(range(len(texts_arg)))
            texts_or_indices = texts_arg
        else:
            selected_records = [records[index_lookup[idx]] for idx in selected_indices]
            texts_or_indices = [r.text for r in selected_records]
            record_names_for_precompute = [str(r.uid) for r in selected_records]
            prior_edits_for_precompute = [
                r.metadata.get("prior_token_edits", []) if isinstance(r.metadata, dict) else []
                for r in selected_records
            ]
            record_positions = selected_indices
        dataset_indices = None
    
    metadata = {
        "dataset": args.data,
        "model": args.model,
        "unique_name": args.unique_name,
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
    if prior_edits_for_precompute is not None:
        pre_kwargs["prior_edits_list"] = prior_edits_for_precompute
    if precompute_config.get("absolute_risk"):
        pre_kwargs["absolute_risk"] = True
    model.pre_stream_anonymize(texts_or_indices=anonymization_inputs, risk_scores=pre_risk_scores, **pre_kwargs)
    pre_elapsed = time.time() - pre_start
    print(f"✓ Pre-computation before anonymization completed in {pre_elapsed:.2f}s")

    run_start = time.time()
    processed = 0

    output_source = source_indices if texts_arg is None else None
    output_indices = map_output_indices(record_positions, output_source)
    stream = model.stream_anonymize(texts_or_indices=anonymization_inputs, buckets=buckets)
    for abs_idx, result_list in zip(output_indices, stream):
        for hp, result in result_list:
            output_handler.output(result, idx=abs_idx, hyperparams=hp, **metadata)
        processed += 1

    total_time = time.time() - run_start
    
    avg_time = total_time / processed if processed > 0 else 0
    print(f"\n{'='*80}\nAnonymization Performance:\n  Total time: {total_time:.2f}s\n  Texts processed: {processed}\n  Average time per text: {avg_time:.2f}s\n  Throughput: {processed/total_time:.2f} texts/s" if total_time > 0 and processed > 0 else "\n  Throughput: N/A")
    print('='*80)
    
    if hasattr(output_handler, 'close'):
        output_handler.close()
