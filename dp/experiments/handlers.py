from __future__ import annotations

import glob
import hashlib
import re
from collections import Counter
from typing import Any, Dict, List, NamedTuple, Optional, Sequence, Set, Tuple

import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedShuffleSplit

from dp.experiments.config import (
    align_evaluation_texts,
    build_utility_target,
    build_vectorizer_from_config,
    ensure_sequence,
    parse_component_config,
    parse_metric_config,
    select_records,
)
from dp.experiments.divergence.base import TextDivergenceExperiment
from dp.experiments.divergence.bertscore import BERTScoreDivergence
from dp.experiments.divergence.cosine import CosineSimilarityDivergence
from dp.experiments.divergence.pp import PerturbationPercentageDivergence
from dp.experiments.divergence.io import (
    build_divergence_evaluation_inputs,
    build_original_texts,
    build_record_info,
)
from dp.experiments.divergence.reporting import build_divergence_report, create_divergence_outputter
from dp.experiments.privacy.io import (
    build_privacy_evaluation_dataset_from_indexed_texts,
    read_indexed_texts_from_jsonl,
)
from dp.experiments.privacy.reporting import build_privacy_report, create_privacy_outputter
from dp.experiments.privacy_annotations import TextPrivacyExperiment
from dp.experiments.utility.base import TextUtilityExperiment
from dp.experiments.utility.utility_runtime import UtilityConfig, run_utility_experiment
from dp.experiments.utility.reporting import build_utility_report, create_utility_outputter
from dp.experiments.utils import build_output_sink, collect_jsonl_sources
from dp.loaders import DatasetRecord, get_adapter
from dp.loaders.derive import get_getter
from dp.utils.tasking import apply_task_template


ConfigDict = Dict[str, Any]
_CHECKPOINT_PATTERN = re.compile(r"checkpoint-(\d+)")


class UtilityCtx(NamedTuple):
    dataset: str
    data_in: str
    split: Optional[str]
    annotations: List[str]
    group_by: Optional[str]
    spec: Any
    selection_criteria: Dict[str, Any]
    debug: bool
    test_size: float
    random_state: int
    dry_run: bool
    output_format: str
    output_file: Optional[str]
    identifier: Optional[str]
    source_splits: List[Dict[str, Any]]
    task_id: Optional[int]


class DivergenceCtx(NamedTuple):
    dataset: str
    data_in: str
    split: Optional[str]
    annotations: List[str]
    metric_type: str
    metric_params: Dict[str, Any]
    max_records: Optional[int]
    output_format: str
    output_file: Optional[str]
    task_id: Optional[int]


class PrivacyCtx(NamedTuple):
    dataset: str
    data_in: str
    split: Optional[str]
    annotations: List[str]
    tri_pipeline: str
    max_records: Optional[int]
    mask_token: str
    tri_max_length: int
    tri_device: str
    progress: bool
    output_format: str
    output_file: Optional[str]
    task_id: Optional[int]


def load_records(dataset: str, data_in: Optional[str], max_records: Optional[int], split: Optional[str] = None) -> List[DatasetRecord]:
    adapter = get_adapter(dataset, data=dataset, data_in=data_in, max_records=max_records, split=split)
    return list(adapter.iter_records())


def _build_index_to_key(dataset: str, data_in: Optional[str], max_records: Optional[int]) -> Dict[int, str]:
    adapter = get_adapter(dataset, data=dataset, data_in=data_in, max_records=max_records)
    return {idx: _record_key(rec, idx) for idx, rec in enumerate(adapter.iter_records())}


def _parse_optional_task_id(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except Exception:
        return None


def _render_runtime_templates(value: Any, task_id: Optional[int]) -> Any:
    if isinstance(value, dict):
        return {k: _render_runtime_templates(v, task_id) for k, v in value.items()}
    if isinstance(value, list):
        return [_render_runtime_templates(v, task_id) for v in value]
    if isinstance(value, tuple):
        return tuple(_render_runtime_templates(v, task_id) for v in value)
    if not isinstance(value, str):
        return value
    rendered = apply_task_template(value, task_id)
    return _resolve_checkpoint_template(rendered)


def merge_params(config: ConfigDict, args: Any) -> ConfigDict:
    out = dict(config)
    identifier = getattr(args, "identifier", None)
    if identifier is not None:
        out["identifier"] = identifier
    task_id = _parse_optional_task_id(out.get("task_id"))
    out = _render_runtime_templates(out, task_id)
    if task_id is not None:
        out["task_id"] = task_id
    return out


def normalize_output_settings(params: ConfigDict) -> None:
    out_cfg = params.get("output")
    if isinstance(out_cfg, dict):
        fmt = out_cfg.get("format")
        path = out_cfg.get("file")
        if fmt and "output_format" not in params:
            params["output_format"] = fmt
        if path and "output_file" not in params:
            params["output_file"] = path


def _require_fields(params: ConfigDict, required: Sequence[str]) -> None:
    for key in required:
        if not params.get(key):
            raise ValueError(f"{key} is required")


def _resolve_annotations(params: ConfigDict) -> List[str]:
    values = ensure_sequence(params.get("annotations")) + ensure_sequence(params.get("annotations_in"))
    return list(dict.fromkeys(values))


def _debug_print_target(spec: Any, params: ConfigDict) -> None:
    payload = params.get("target")
    print(f"Utility target: dataset={spec.dataset} key={spec.target_key} mode={spec.target.mode.value}")
    if isinstance(payload, dict):
        enum_vals = ensure_sequence(payload.get("enum"))
        if enum_vals:
            print(f"Target enum: {enum_vals}")


def _debug_print_records(spec: Any, records: Sequence[DatasetRecord]) -> None:
    values = [spec.target.value(r) for r in records]
    coverage = sum(1 for v in values if v is not None)
    dist = Counter([str(v) for v in values if v is not None])
    print(f"Records loaded: {len(records)}")
    print(f"Target coverage: {coverage}")
    print(f"Target distribution: {dict(dist)}")


def _resolve_utility_components(spec: Any, params: ConfigDict) -> Tuple[str | None, Dict[str, Any], str | None, Dict[str, Any]]:
    vec_name, vec_kwargs = parse_component_config(params.get("vectorizer"))
    head_name, head_kwargs = parse_component_config(params.get("head"))
    if not head_name or not vec_name:
        from dp.experiments.utility.models import MODE_TO_MODEL

        desired = MODE_TO_MODEL.get(spec.target.mode)
        if desired:
            if not vec_name:
                vec_name = desired[0]
            if not head_name:
                head_name = desired[1]
    return vec_name, vec_kwargs, head_name, head_kwargs


def _resolve_checkpoint_template(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _resolve_checkpoint_template(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_resolve_checkpoint_template(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_resolve_checkpoint_template(v) for v in value)
    if not isinstance(value, str):
        return value
    if "{checkpoint}" not in value:
        return value
    pattern = value.replace("{checkpoint}", "checkpoint-*")
    matches = sorted(glob.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"no checkpoint matched pattern: {pattern}")
    best: Optional[Tuple[int, str]] = None
    for candidate in matches:
        found = _CHECKPOINT_PATTERN.search(candidate)
        if not found:
            continue
        step = int(found.group(1))
        if best is None or step > best[0]:
            best = (step, candidate)
    if best is None:
        raise FileNotFoundError(f"no checkpoint-* entry matched pattern: {pattern}")
    return best[1]


def _render_component_paths(value: Any, task_id: Optional[int]) -> Any:
    if isinstance(value, dict):
        return {k: _render_component_paths(v, task_id) for k, v in value.items()}
    if isinstance(value, list):
        return [_render_component_paths(v, task_id) for v in value]
    if isinstance(value, tuple):
        return tuple(_render_component_paths(v, task_id) for v in value)
    if not isinstance(value, str):
        return value
    rendered = apply_task_template(value, task_id)
    return _resolve_checkpoint_template(rendered)


def _debug_print_components(vec_name: str | None, vec_kwargs: Dict[str, Any], head_name: str | None, head_kwargs: Dict[str, Any]) -> None:
    print(f"Vectorizer: name={vec_name or 'auto'} params={vec_kwargs}")
    print(f"Head: name={head_name or 'auto'} params={head_kwargs}")


def _normalize_target_value(value: Any, mode: Any) -> Optional[Any]:
    if value is None:
        return None
    if str(mode.value) == "cardinal":
        try:
            return float(value)
        except Exception:
            return None
    s = str(value).strip()
    if not s:
        return None
    return s


def _prepare_utility(params: ConfigDict) -> UtilityCtx:
    annotations = _resolve_utility_annotations(params)
    _require_fields(params, ["dataset", "data_in", "target"])
    task_id = _parse_optional_task_id(params.get("task_id"))
    dataset = str(params.get("dataset"))
    data_in = str(apply_task_template(str(params.get("data_in")), task_id))
    split_value = params.get("split")
    split: Optional[str]
    if isinstance(split_value, str):
        split = apply_task_template(split_value, task_id)
    else:
        split = None
    source_splits = params.get("source_splits")
    if source_splits is None and isinstance(params.get("split"), list):
        source_splits = params.get("split")
    source_splits = source_splits if isinstance(source_splits, list) else []
    group_by = params.get("group_by")
    spec = build_utility_target(params, dataset)
    selection_criteria = params.get("selection_criteria", {}) or {}
    debug = bool(params.get("debug", False))
    test_size = float(params.get("test_size", 0.0))
    random_state = int(params.get("random_state", 42))
    dry_run = bool(params.get("dry_run", False))
    output_format = str(params.get("output_format", "text"))
    output_file = params.get("output_file")
    if isinstance(source_splits, list) and source_splits:
        main_split = next((s for s in source_splits if "path" not in s and "split" not in s), None)
        if isinstance(main_split, dict) and main_split.get("val") is not None:
            try:
                test_size = float(main_split.get("val"))
            except Exception:
                pass
    identifier = params.get("identifier")
    return UtilityCtx(
        dataset,
        data_in,
        split,
        annotations,
        group_by,
        spec,
        selection_criteria,
        debug,
        test_size,
        random_state,
        dry_run,
        output_format,
        output_file,
        identifier,
        source_splits,
        task_id,
    )


def _prepare_divergence(params: ConfigDict) -> DivergenceCtx:
    annotations = _resolve_annotations(params)
    _require_fields(params, ["dataset", "data_in"])
    task_id = _parse_optional_task_id(params.get("task_id"))
    dataset = str(params.get("dataset"))
    data_in = str(apply_task_template(str(params.get("data_in")), task_id))
    split_value = params.get("split")
    split = apply_task_template(split_value, task_id) if isinstance(split_value, str) else None
    metric_type, metric_params = parse_metric_config(params.get("metric"))
    max_records = params.get("max_records")
    output_format = str(params.get("output_format", "text"))
    output_file = params.get("output_file")
    return DivergenceCtx(dataset, data_in, split, annotations, metric_type, metric_params, max_records, output_format, output_file, task_id)


def _prepare_privacy(params: ConfigDict) -> PrivacyCtx:
    tri_cfg = params.get("tri")
    if isinstance(tri_cfg, dict):
        if "universal_tri_pipeline" in tri_cfg and "universal_tri_pipeline" not in params:
            params["universal_tri_pipeline"] = tri_cfg["universal_tri_pipeline"]
        if "max_length" in tri_cfg and "tri_max_length" not in params:
            params["tri_max_length"] = tri_cfg["max_length"]
        if "device" in tri_cfg and "tri_device" not in params:
            params["tri_device"] = tri_cfg["device"]
    annotations = _resolve_annotations(params)
    _require_fields(params, ["dataset", "data_in", "universal_tri_pipeline"])
    task_id = _parse_optional_task_id(params.get("task_id"))
    dataset = str(params.get("dataset"))
    data_in = str(apply_task_template(str(params.get("data_in")), task_id))
    split_value = params.get("split")
    split = apply_task_template(split_value, task_id) if isinstance(split_value, str) else None
    tri_pipeline = str(apply_task_template(str(params.get("universal_tri_pipeline")), task_id))
    max_records = params.get("max_records")
    mask_token = str(params.get("mask_token", "[MASK]"))
    tri_max_length = int(params.get("tri_max_length", 512))
    tri_device = params.get("tri_device")
    if params.get("no_progress"):
        progress = False
    elif "progress" in params:
        progress = bool(params.get("progress"))
    else:
        progress = True
    output_format = str(params.get("output_format", "text"))
    output_file = params.get("output_file")
    return PrivacyCtx(dataset, data_in, split, annotations, tri_pipeline, max_records, mask_token, tri_max_length, tri_device, bool(progress), output_format, output_file, task_id)


def build_divergence_experiment(metric_type: str, metric_params: Dict[str, Any]) -> TextDivergenceExperiment:
    if not metric_type:
        raise ValueError("divergence metric type is required")
    if metric_type == "bertscore":
        allowed = {"model_type", "language", "batch_size", "device", "rescale_with_baseline"}
        kwargs = {key: metric_params[key] for key in allowed if key in metric_params}
        return BERTScoreDivergence(**kwargs)
    if metric_type == "cosine":
        vectorizer_config = metric_params["vectorizer"] if "vectorizer" in metric_params else None
        vectorizer = build_vectorizer_from_config(vectorizer_config) if vectorizer_config is not None else None
        return CosineSimilarityDivergence(vectorizer=vectorizer)
    if metric_type == "pp":
        return PerturbationPercentageDivergence()
    raise ValueError(f"unsupported divergence metric '{metric_type}'")


def map_record_key_to_group_label(records: List[DatasetRecord], dataset: str, group_by: str) -> Dict[str, Any]:
    group_getter = get_getter(dataset, group_by)
    mapping: Dict[str, Any] = {}
    for index, record in enumerate(records):
        key = record.uid or f"record_{index + 1}"
        mapping[key] = group_getter(record)
    return mapping


def _record_key(record: DatasetRecord, index: int) -> str:
    if record.uid:
        return str(record.uid)
    if record.name:
        return str(record.name)
    return f"record_{index + 1}"


def _dedupe_records(records: Sequence[DatasetRecord]) -> List[DatasetRecord]:
    seen: Set[str] = set()
    out: List[DatasetRecord] = []
    for idx, record in enumerate(records):
        key = _record_key(record, idx)
        if key in seen:
            continue
        seen.add(key)
        out.append(record)
    return out


def _resolve_split_keys(ctx: UtilityCtx, params: ConfigDict) -> Tuple[Dict[str, List[str]], List[DatasetRecord]]:
    split_cfg = ctx.source_splits if isinstance(ctx.source_splits, list) else []
    role_records: Dict[str, List[DatasetRecord]] = {"train": [], "val": [], "test": []}
    loaded_union: List[DatasetRecord] = []

    for entry in split_cfg:
        if not isinstance(entry, dict):
            continue
        source_path = entry.get("path") or ctx.data_in
        if isinstance(source_path, str):
            source_path = apply_task_template(source_path, ctx.task_id)
        source_split = entry.get("split")
        if isinstance(source_split, str):
            source_split = apply_task_template(source_split, ctx.task_id)
        weights = {
            "train": float(entry.get("train", 0.0) or 0.0),
            "val": float(entry.get("val", 0.0) or 0.0),
            "test": float(entry.get("test", 0.0) or 0.0),
        }
        active = [role for role, weight in weights.items() if weight > 0.0]
        if not active:
            continue
        role = max(active, key=lambda name: weights[name])
        source_records = load_records(ctx.dataset, source_path, params.get("max_records"), split=source_split)
        source_records = select_records(source_records, ctx.selection_criteria)
        if not source_records:
            continue
        role_records[role].extend(source_records)
        loaded_union.extend(source_records)

    if not role_records["test"] and ctx.split:
        test_records = load_records(ctx.dataset, ctx.data_in, params.get("max_records"), split=ctx.split)
        test_records = select_records(test_records, ctx.selection_criteria)
        role_records["test"].extend(test_records)
        loaded_union.extend(test_records)

    if not role_records["test"] and role_records["val"]:
        role_records["test"] = list(role_records["val"])

    deduped_union = _dedupe_records(loaded_union)
    if not deduped_union:
        fallback = load_records(ctx.dataset, ctx.data_in, params.get("max_records"), split=ctx.split)
        fallback = select_records(fallback, ctx.selection_criteria)
        deduped_union = _dedupe_records(fallback)

    role_keys: Dict[str, List[str]] = {"train": [], "val": [], "test": []}
    for role in ("train", "val", "test"):
        deduped = _dedupe_records(role_records[role])
        keys = [_record_key(record, idx) for idx, record in enumerate(deduped)]
        role_keys[role] = keys
    return role_keys, deduped_union


def handle_utility(args: Any, config: ConfigDict) -> None:
    params = merge_params(config, args)
    normalize_output_settings(params)
    ctx = _prepare_utility(params)
    if not ctx.annotations:
        raise ValueError("annotations are required")
    include_cm = bool(params.get("include_confusion_matrix", False))
    records = load_records(ctx.dataset, ctx.data_in, params.get("max_records"), split=ctx.split)
    maybe_group_mapping = None
    if ctx.debug:
        _debug_print_target(ctx.spec, params)
    records = select_records(records, ctx.selection_criteria)
    if not records:
        raise RuntimeError("no records selected by criteria")
    if ctx.group_by:
        maybe_group_mapping = map_record_key_to_group_label(records, ctx.dataset, ctx.group_by)
    if ctx.debug:
        _debug_print_records(ctx.spec, records)
    if ctx.dry_run:
        coverage = sum(1 for record in records if ctx.spec.target.value(record) is not None and record.text)
        print(f"Records loaded: {len(records)}")
        print(f"Target coverage: {coverage}")
        return
    sources = collect_jsonl_sources(*ctx.annotations, task_id=ctx.task_id)
    if not sources:
        raise RuntimeError("no anonymized output files discovered")
    split_keys, split_records = _resolve_split_keys(ctx, params)
    records = split_records or records
    index_to_key = _build_index_to_key(ctx.dataset, ctx.data_in, params.get("max_records"))
    protocol = str(params.get("protocol", "utility")).strip().lower() or "utility"
    evaluation_texts = None
    if protocol not in {"utility", "supervised_divergence"}:
        evaluation_texts = align_evaluation_texts(index_to_key, records, sources)
    if ctx.debug:
        print("Evaluation sources:")
        if evaluation_texts is None:
            for name, path in sorted(sources.items(), key=lambda item: item[0]):
                print(f"- {name}: path={path}")
        else:
            for name, mapping in evaluation_texts.items():
                count = len(mapping)
                sample = next(iter(mapping.values()), "")
                sig = hashlib.sha1(sample.encode("utf-8")).hexdigest()[:16] if sample else ""
                print(f"- {name}: count={count} sample_sig={sig}")
    if evaluation_texts is not None and not evaluation_texts:
        raise RuntimeError("no anonymized texts aligned with dataset records")
    vec_name, vec_kwargs, head_name, head_kwargs = _resolve_utility_components(ctx.spec, params)
    vec_kwargs = _render_component_paths(vec_kwargs, ctx.task_id)
    head_kwargs = _render_component_paths(head_kwargs, ctx.task_id)
    if ctx.debug:
        _debug_print_components(vec_name, vec_kwargs, head_name, head_kwargs)
    vectorizer, model = ctx.spec.build_components(
        vectorizer_name=vec_name or None,
        vectorizer_kwargs=vec_kwargs,
        head_name=head_name or None,
        head_kwargs=head_kwargs,
        identifier=ctx.identifier,
    )
    resolved_model_name = str(getattr(model, "name", head_name or ctx.spec.default_head))
    resolved_primary_metric = str(getattr(model, "primary_metric", head_kwargs.get("primary_metric", "")))
    if protocol in {"utility", "supervised_divergence"}:
        model.cleanup()
        vectorizer.cleanup()
        utility_cfg_raw = params.get("utility", {}) or {}
        if not isinstance(utility_cfg_raw, dict):
            raise ValueError("utility config must be a mapping")
        tune_cfg = utility_cfg_raw.get("tune", {}) or {}
        if not isinstance(tune_cfg, dict):
            raise ValueError("utility.tune must be a mapping")
        tune_key_value = tune_cfg.get("key")
        tune_values_raw = tune_cfg.get("values")
        tune_values = ensure_sequence(tune_values_raw)
        tune_key = str(tune_key_value).strip() if tune_key_value is not None else None
        if tune_key == "":
            tune_key = None
        utility_cfg = UtilityConfig(
            random_state=int(utility_cfg_raw.get("random_state", ctx.random_state)),
            tune_param=tune_key,
            tune_values=tune_values,
        )
        result = run_utility_experiment(
            spec=ctx.spec,
            records=records,
            evaluation_texts=evaluation_texts,
            evaluation_sources=sources,
            index_to_key=index_to_key,
            vectorizer_name=vec_name or None,
            vectorizer_kwargs=vec_kwargs,
            head_name=head_name or None,
            head_kwargs=head_kwargs,
            identifier=ctx.identifier,
            config=utility_cfg,
            model_name=resolved_model_name,
            primary_metric=resolved_primary_metric,
            split_keys=split_keys,
            protocol=protocol,
        )
        report = build_utility_report(result, sources)
        sink = build_output_sink(ctx.output_file)
        outputter = create_utility_outputter(ctx.output_format, sink)
        outputter.output(report)
        return
    split_cfg = ctx.source_splits
    train_texts_override: List[str] = []
    train_labels_override: List[Any] = []
    test_texts_override: List[str] = []
    test_labels_override: List[Any] = []
    train_uids: List[str] = []
    test_uids: List[str] = []

    if isinstance(split_cfg, list):
        for entry in split_cfg:
            if not isinstance(entry, dict):
                continue
            source_path = entry.get("path") or ctx.data_in
            if isinstance(source_path, str):
                source_path = apply_task_template(source_path, ctx.task_id)
            source_split = entry.get("split")
            if isinstance(source_split, str):
                source_split = apply_task_template(source_split, ctx.task_id)
            train_ratio = float(entry.get("train", 0))
            val_ratio = float(entry.get("val", 0))
            if train_ratio + val_ratio > 1:
                raise ValueError(f"train + val ratios must be <= 1, got {train_ratio} + {val_ratio}")
            if train_ratio < 0 or val_ratio < 0:
                raise ValueError("ratios must be >= 0")

            source_records = load_records(ctx.dataset, source_path, params.get("max_records"), split=source_split)
            source_records = select_records(source_records, ctx.selection_criteria)
            if not source_records:
                continue

            labels_for_split = []
            valid_indices = []
            for idx, rec in enumerate(source_records):
                v = ctx.spec.target.value(rec)
                if v is None or not rec.text:
                    continue
                nv = _normalize_target_value(v, ctx.spec.target.mode)
                if nv is None:
                    continue
                labels_for_split.append(nv)
                valid_indices.append(idx)

            if not labels_for_split:
                continue

            labels_array = np.array(labels_for_split)
            valid_indices_array = np.array(valid_indices)

            if train_ratio > 0 and val_ratio > 0:
                splitter = StratifiedShuffleSplit(
                    n_splits=1,
                    test_size=val_ratio / (train_ratio + val_ratio),
                    random_state=ctx.random_state,
                )
                train_mask, val_mask = next(splitter.split(valid_indices_array, labels_array))
                train_indices = valid_indices_array[train_mask]
                val_indices = valid_indices_array[val_mask]
            elif train_ratio > 0:
                train_count = int(len(valid_indices_array) * train_ratio)
                train_indices = valid_indices_array[:train_count]
                val_indices = np.array([], dtype=int)
            elif val_ratio > 0:
                val_count = int(len(valid_indices_array) * val_ratio)
                val_indices = valid_indices_array[:val_count]
                train_indices = np.array([], dtype=int)
            else:
                train_indices = np.array([], dtype=int)
                val_indices = np.array([], dtype=int)

            for idx in train_indices:
                rec = source_records[int(idx)]
                v = ctx.spec.target.value(rec)
                nv = _normalize_target_value(v, ctx.spec.target.mode)
                train_texts_override.append(rec.text)
                train_labels_override.append(nv)
                train_uids.append(rec.uid or f"record_{idx}")

            for idx in val_indices:
                rec = source_records[int(idx)]
                v = ctx.spec.target.value(rec)
                nv = _normalize_target_value(v, ctx.spec.target.mode)
                test_texts_override.append(rec.text)
                test_labels_override.append(nv)
                test_uids.append(rec.uid or f"record_{idx}")

    if train_uids and test_uids:
        test_uid_set = set(test_uids)
        filtered_train_texts: List[str] = []
        filtered_train_labels: List[Any] = []
        filtered_train_uids: List[str] = []
        for text, label, uid in zip(train_texts_override, train_labels_override, train_uids):
            if uid in test_uid_set:
                continue
            filtered_train_texts.append(text)
            filtered_train_labels.append(label)
            filtered_train_uids.append(uid)
        train_texts_override = filtered_train_texts
        train_labels_override = filtered_train_labels
        train_uids = filtered_train_uids

    if ctx.debug:
        original_uids = [rec.uid or f"record_{i + 1}" for i, rec in enumerate(records)]
        print("\n=== Dataset Loading Summary ===")
        print(f"Original dataset (data_in={ctx.data_in}): {len(records)} records")
        print(f"Original UIDs (first 10): {original_uids[:10]}")
        if train_texts_override:
            print(f"\nTraining set: {len(train_texts_override)} records")
            print(f"Training UIDs (first 10): {train_uids[:10]}")
        if test_texts_override:
            print(f"\nValidation set: {len(test_texts_override)} records")
            print(f"Validation UIDs (first 10): {test_uids[:10]}")
        if train_uids and test_uids:
            overlap = set(train_uids) & set(test_uids)
            if overlap:
                print(f"\nWARNING: Overlap between train and val: {len(overlap)} records")
            else:
                print("\nNo overlap between train and val: disjoint sets")

    experiment = TextUtilityExperiment(
        test_size=ctx.test_size,
        random_state=ctx.random_state,
        include_confusion_matrix=include_cm,
    )
    experiment.setup(
        target=ctx.spec.target,
        records=records,
        vectorizer=vectorizer,
        model=model,
        train_texts_override=train_texts_override or None,
        train_labels_override=train_labels_override or None,
        test_texts_override=test_texts_override or None,
        test_labels_override=test_labels_override or None,
        maybe_group_mapping=maybe_group_mapping,
        group_by=ctx.group_by,
    )
    if ctx.debug:
        train_sz = len(getattr(experiment, "_train_keys", []))
        test_sz = len(getattr(experiment, "_test_keys", []))
        if train_texts_override:
            print(f"Training override size: {len(train_texts_override)}")
        x_train = getattr(experiment, "_x_train", None)
        lbls_train = getattr(experiment, "_train_labels", [])
        lbls_test = getattr(experiment, "_test_labels", [])
        print(f"Eval split sizes: train={train_sz} test={test_sz}")
        try:
            desc = vectorizer.describe()
            print(f"Vectorizer describe: {desc}")
        except Exception:
            pass
        if x_train is not None:
            try:
                arr = np.asarray(x_train)
            except Exception:
                arr = x_train
            if hasattr(arr, "shape"):
                print(f"X_train shape: {arr.shape}")
            try:
                vals = arr.toarray() if hasattr(arr, "toarray") else np.asarray(arr)
                print(f"X_train stats: mean={float(np.nanmean(vals)):.6f} std={float(np.nanstd(vals)):.6f}")
            except Exception:
                pass
        try:
            enc = getattr(model, "_label_encoder", None)
            if enc is not None and hasattr(enc, "classes_"):
                print(f"Label classes: {list(enc.classes_)}")
        except Exception:
            pass
        if train_texts_override and train_labels_override:
            try:
                vec = getattr(experiment, "_vectorizer", None)
                mdl = getattr(experiment, "_model", None)
                x_train_override = vec.transform(train_texts_override)
                preds_train_override = mdl.predict(x_train_override)
                dist_preds_train = Counter([str(p) for p in preds_train_override])
                dist_true_train = Counter([str(y) for y in train_labels_override])
                print("\n=== Training Dataset (Synthetic Override) ===")
                print(f"Predictions distribution: {dict(dist_preds_train)}")
                print(f"True distribution: {dict(dist_true_train)}")
                try:
                    all_labels_set = getattr(mdl, "_label_order", None) or sorted(list(set(train_labels_override)))
                    cm_train = confusion_matrix(train_labels_override, preds_train_override, labels=all_labels_set)
                    print(f"Confusion matrix shape: {cm_train.shape}")
                    print(cm_train)
                except Exception:
                    pass
                eval_train = mdl.evaluate(x_train_override, train_labels_override)
                print(f"Evaluation metrics: {eval_train}")
                import torch

                if torch.backends.mps.is_available():
                    torch.mps.empty_cache()
            except Exception:
                pass
        if test_sz > 0:
            try:
                x_test_dbg = getattr(experiment, "_vectorizer", None).transform(getattr(experiment, "_test_texts", []))
                preds_dbg = getattr(experiment, "_model", None).predict(x_test_dbg)
                dist_preds = Counter([str(p) for p in preds_dbg])
                dist_true = Counter([str(y) for y in lbls_test])
                print("\n=== Validation Dataset (Original Records) ===")
                print(f"Predictions distribution: {dict(dist_preds)}")
                print(f"True distribution: {dict(dist_true)}")
                try:
                    mdl_val = getattr(experiment, "_model", None)
                    all_labels_set = getattr(mdl_val, "_label_order", None) or sorted(list(set(lbls_test)))
                    cm = confusion_matrix(lbls_test, preds_dbg, labels=all_labels_set)
                    print(f"Confusion matrix shape: {cm.shape}")
                    print(cm)
                except Exception:
                    pass
                eval_metrics = getattr(experiment, "_model", None).evaluate(x_test_dbg, lbls_test)
                print(f"Evaluation metrics: {eval_metrics}")
                import torch

                if torch.backends.mps.is_available():
                    torch.mps.empty_cache()
            except Exception:
                pass
        try:
            all_texts_dbg = [rec.text for rec in getattr(experiment, "_records", [])]
            lbls_all = list(getattr(experiment, "_labels", []))
            x_all_dbg = getattr(experiment, "_vectorizer", None).transform(all_texts_dbg)
            preds_all = getattr(experiment, "_model", None).predict(x_all_dbg)
            dist_preds_all = Counter([str(p) for p in preds_all])
            dist_true_all = Counter([str(y) for y in lbls_all])
            print("\n=== Overall Dataset (Full Original Records) ===")
            print(f"Predictions distribution: {dict(dist_preds_all)}")
            print(f"True distribution: {dict(dist_true_all)}")
            try:
                mdl_all = getattr(experiment, "_model", None)
                cm_all = confusion_matrix(
                    lbls_all,
                    preds_all,
                    labels=getattr(mdl_all, "_label_order", None) or sorted(list(set(lbls_all))),
                )
                print(f"Full confusion matrix shape: {cm_all.shape}")
                print(cm_all)
            except Exception:
                pass
            eval_all = getattr(experiment, "_model", None).evaluate(x_all_dbg, lbls_all)
            print(f"Evaluation metrics: {eval_all}")
            import torch

            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
        except Exception:
            pass
    result = experiment.run(evaluation_texts=evaluation_texts)
    experiment.cleanup()
    report = build_utility_report(result, sources)
    sink = build_output_sink(ctx.output_file)
    outputter = create_utility_outputter(ctx.output_format, sink)
    outputter.output(report)


def handle_divergence(args: Any, config: ConfigDict) -> None:
    params = merge_params(config, args)
    normalize_output_settings(params)
    ctx = _prepare_divergence(params)
    if not ctx.annotations:
        raise ValueError("annotations are required")
    records = load_records(ctx.dataset, ctx.data_in, ctx.max_records, split=ctx.split)
    if not records:
        raise RuntimeError("no records loaded")
    sources = collect_jsonl_sources(*ctx.annotations, task_id=ctx.task_id)
    if not sources:
        raise RuntimeError("no anonymized output files discovered")
    div_index_to_key = _build_index_to_key(ctx.dataset, ctx.data_in, ctx.max_records)
    evaluation_inputs = build_divergence_evaluation_inputs(div_index_to_key, sources)
    evaluation_inputs = {name: payload for name, payload in evaluation_inputs.items() if payload.get("texts")}
    if not evaluation_inputs:
        raise RuntimeError("no anonymized outputs aligned with dataset records")
    record_info = build_record_info(records)
    original_texts = build_original_texts(records)
    experiment = build_divergence_experiment(ctx.metric_type, ctx.metric_params)
    experiment.setup(
        original_texts=original_texts,
        evaluation_datasets=evaluation_inputs,
        record_info=record_info,
    )
    result = experiment.run()
    experiment.cleanup()
    report = build_divergence_report(result, len(records), sources)
    sink = build_output_sink(ctx.output_file)
    outputter = create_divergence_outputter(ctx.output_format, sink)
    outputter.output(report)


def handle_privacy(args: Any, config: ConfigDict) -> None:
    params = merge_params(config, args)
    normalize_output_settings(params)
    ctx = _prepare_privacy(params)
    if not ctx.annotations:
        raise ValueError("annotations are required")
    records = load_records(ctx.dataset, ctx.data_in, ctx.max_records, split=ctx.split)
    if not records:
        raise RuntimeError("no records loaded")
    reference_records = load_records(ctx.dataset, ctx.data_in, ctx.max_records, split=None)
    if not reference_records:
        raise RuntimeError("no reference records loaded")
    sources = collect_jsonl_sources(*ctx.annotations, task_id=ctx.task_id)
    if not sources:
        raise RuntimeError("no annotation files discovered")
    evaluation_datasets: Dict[str, List[DatasetRecord]] = {}
    evaluation_counts: Dict[str, int] = {}
    for name, path in sources.items():
        indexed_texts = read_indexed_texts_from_jsonl(path)
        if len(indexed_texts) == len(records):
            alignment_mode = "direct"
        elif len(indexed_texts) == len(reference_records):
            alignment_mode = "projected-from-full"
        else:
            alignment_mode = "invalid"
        print(
            "Privacy alignment "
            f"source={name} mode={alignment_mode} "
            f"rows={len(indexed_texts)} split_records={len(records)} full_records={len(reference_records)}"
        )
        dataset_records = build_privacy_evaluation_dataset_from_indexed_texts(
            records,
            indexed_texts,
            reference_records=reference_records,
        )
        evaluation_datasets[name] = dataset_records
        evaluation_counts[name] = len(dataset_records)
    experiment = TextPrivacyExperiment(
        tri_pipeline=ctx.tri_pipeline,
        tri_max_length=ctx.tri_max_length,
        tri_device=ctx.tri_device,
    )
    experiment.setup(
        dataset_name=ctx.dataset,
        original_dataset=records,
        evaluation_datasets=evaluation_datasets,
        progress=ctx.progress,
    )
    result = experiment.run(progress=ctx.progress)
    experiment.cleanup()
    report = build_privacy_report(
        result=result,
        original_record_count=len(records),
        annotation_sources=sources,
        evaluation_record_counts=evaluation_counts,
    )
    sink = build_output_sink(ctx.output_file)
    outputter = create_privacy_outputter(ctx.output_format, sink)
    outputter.output(report)


def _resolve_utility_annotations(params: ConfigDict) -> List[str]:
    return _resolve_annotations(params)


HANDLERS: Dict[str, Any] = {
    "utility": handle_utility,
    "divergence": handle_divergence,
    "privacy": handle_privacy,
}
