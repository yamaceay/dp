from __future__ import annotations

import hashlib
from collections import Counter
from typing import Any, Dict, List, Optional, Sequence, Tuple, NamedTuple

import numpy as np
from sklearn.metrics import confusion_matrix

from dp.experiments.divergence.io import (
    build_divergence_evaluation_inputs,
    build_original_texts,
    build_record_info,
)
from dp.experiments.divergence.reporting import build_divergence_report, create_divergence_outputter
from dp.experiments.privacy.io import (
    build_privacy_evaluation_dataset_from_texts,
    read_texts_from_jsonl,
)
from dp.experiments.privacy.reporting import build_privacy_report, create_privacy_outputter
from dp.experiments.utility.reporting import build_utility_report, create_utility_outputter
from dp.experiments.divergence.bertscore import BERTScoreDivergence
from dp.experiments.divergence.cosine import CosineSimilarityDivergence
from dp.experiments.divergence.base import TextDivergenceExperiment
from dp.experiments.privacy_annotations import TextPrivacyExperiment
from dp.experiments.utility.base import TextUtilityExperiment
from dp.experiments.utils import build_output_sink, collect_jsonl_sources
from dp.loaders import DatasetRecord, get_adapter

from runtime.runner_helpers import (
    ensure_sequence,
    parse_component_config,
    parse_metric_config,
    build_vectorizer_from_config,
    select_records,
    align_evaluation_texts,
    build_utility_target,
)


ConfigDict = Dict[str, Any]


class UtilityCtx(NamedTuple):
    dataset: str
    data_in: str
    annotations: List[str]
    spec: Any
    selection_criteria: Dict[str, Any]
    debug: bool
    test_size: float
    random_state: int
    dry_run: bool
    output_format: str
    output_file: Optional[str]
    identifier: Optional[str]


class DivergenceCtx(NamedTuple):
    dataset: str
    data_in: str
    annotations: List[str]
    metric_type: str
    metric_params: Dict[str, Any]
    max_records: Optional[int]
    output_format: str
    output_file: Optional[str]


class PrivacyCtx(NamedTuple):
    dataset: str
    data_in: str
    annotations: List[str]
    tri_pipeline: str
    max_records: Optional[int]
    mask_token: str
    tri_max_length: int
    tri_device: str
    progress: bool
    output_format: str
    output_file: Optional[str]


def load_records(dataset: str, data_in: Optional[str], max_records: Optional[int]) -> List[DatasetRecord]:
    adapter = get_adapter(dataset, data=dataset, data_in=data_in, max_records=max_records)
    return list(adapter.iter_records())


def merge_params(config: ConfigDict, args: Any) -> ConfigDict:
    out = dict(config)
    identifier = getattr(args, "identifier", None)
    if identifier is not None:
        out["identifier"] = identifier
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


def _debug_print_components(vec_name: str | None, vec_kwargs: Dict[str, Any], head_name: str | None, head_kwargs: Dict[str, Any]) -> None:
    print(f"Vectorizer: name={vec_name or 'auto'} params={vec_kwargs}")
    print(f"Head: name={head_name or 'auto'} params={head_kwargs}")


def _prepare_utility(params: ConfigDict) -> UtilityCtx:
    annotations = _resolve_utility_annotations(params)
    _require_fields(params, ["dataset", "data_in", "target"]) 
    dataset = str(params.get("dataset"))
    data_in = str(params.get("data_in"))
    spec = build_utility_target(params, dataset)
    selection_criteria = params.get("selection_criteria", {}) or {}
    debug = bool(params.get("debug", False))
    test_size = float(params.get("test_size", 0.0))
    random_state = int(params.get("random_state", 42))
    dry_run = bool(params.get("dry_run", False))
    output_format = str(params.get("output_format", "text"))
    output_file = params.get("output_file")
    split_cfg = params.get("split")
    if isinstance(split_cfg, list) and split_cfg:
        main_split = next((s for s in split_cfg if "path" not in s), None)
        if isinstance(main_split, dict) and main_split.get("val") is not None:
            try:
                test_size = float(main_split.get("val"))
            except Exception:
                pass
    identifier = params.get("identifier")
    return UtilityCtx(dataset, data_in, annotations, spec, selection_criteria, debug, test_size, random_state, dry_run, output_format, output_file, identifier)


def _prepare_divergence(params: ConfigDict) -> DivergenceCtx:
    annotations = _resolve_annotations(params)
    _require_fields(params, ["dataset", "data_in"]) 
    dataset = str(params.get("dataset"))
    data_in = str(params.get("data_in"))
    metric_type, metric_params = parse_metric_config(params.get("metric"))
    max_records = params.get("max_records")
    output_format = str(params.get("output_format", "text"))
    output_file = params.get("output_file")
    return DivergenceCtx(dataset, data_in, annotations, metric_type, metric_params, max_records, output_format, output_file)


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
    dataset = str(params.get("dataset"))
    data_in = str(params.get("data_in"))
    tri_pipeline = str(params.get("universal_tri_pipeline"))
    max_records = params.get("max_records")
    mask_token = str(params.get("mask_token", "[MASK]"))
    tri_max_length = int(params.get("tri_max_length", 512))
    tri_device = str(params.get("tri_device", "auto"))
    if params.get("no_progress"):
        progress = False
    elif "progress" in params:
        progress = bool(params.get("progress"))
    else:
        progress = True
    output_format = str(params.get("output_format", "text"))
    output_file = params.get("output_file")
    return PrivacyCtx(dataset, data_in, annotations, tri_pipeline, max_records, mask_token, tri_max_length, tri_device, bool(progress), output_format, output_file)


def build_divergence_experiment(metric_type: str, metric_params: Dict[str, Any]) -> TextDivergenceExperiment:
    if metric_type == "bertscore":
        allowed = {"model_type", "language", "batch_size", "device", "rescale_with_baseline"}
        kwargs = {key: metric_params[key] for key in allowed if key in metric_params}
        return BERTScoreDivergence(**kwargs)
    if metric_type == "cosine":
        vectorizer = build_vectorizer_from_config(metric_params.get("vectorizer"))
        return CosineSimilarityDivergence(vectorizer=vectorizer)
    raise ValueError(f"unsupported divergence metric '{metric_type}'")


def handle_utility(args: Any, config: ConfigDict) -> None:
    params = merge_params(config, args)
    normalize_output_settings(params)
    ctx = _prepare_utility(params)
    if not ctx.annotations:
        raise ValueError("annotations are required")
    records = load_records(ctx.dataset, ctx.data_in, params.get("max_records"))
    if ctx.debug:
        _debug_print_target(ctx.spec, params)
    records = select_records(records, ctx.selection_criteria)
    if not records:
        raise RuntimeError("no records selected by criteria")
    if ctx.debug:
        _debug_print_records(ctx.spec, records)
    if ctx.dry_run:
        coverage = sum(1 for record in records if ctx.spec.target.value(record) is not None and record.text)
        print(f"Records loaded: {len(records)}")
        print(f"Target coverage: {coverage}")
        return
    sources = collect_jsonl_sources(*ctx.annotations)
    if not sources:
        raise RuntimeError("no anonymized output files discovered")
    evaluation_texts = align_evaluation_texts(records, sources)
    if ctx.debug:
        print("Evaluation sources:")
        for name, mapping in evaluation_texts.items():
            count = len(mapping)
            sample = next(iter(mapping.values()), "")
            sig = hashlib.sha1(sample.encode("utf-8")).hexdigest()[:16] if sample else ""
            print(f"- {name}: count={count} sample_sig={sig}")
    if not evaluation_texts:
        raise RuntimeError("no anonymized texts aligned with dataset records")
    vec_name, vec_kwargs, head_name, head_kwargs = _resolve_utility_components(ctx.spec, params)
    if ctx.debug:
        _debug_print_components(vec_name, vec_kwargs, head_name, head_kwargs)
    vectorizer, model = ctx.spec.build_components(
        vectorizer_name=vec_name or None,
        vectorizer_kwargs=vec_kwargs,
        head_name=head_name or None,
        head_kwargs=head_kwargs,
        identifier=ctx.identifier,
    )
    split_cfg = params.get("split")
    extra_path = None
    if isinstance(split_cfg, list):
        extra_entry = next((s for s in split_cfg if isinstance(s, dict) and s.get("path") and float(s.get("train", 0)) > 0), None)
        if extra_entry:
            extra_path = str(extra_entry.get("path"))
    
    train_texts_override: List[str] = []
    train_labels_override: List[Any] = []
    if extra_path:
        extra_records = load_records(ctx.dataset, extra_path, params.get("max_records"))
        extra_records = select_records(extra_records, ctx.selection_criteria)
        if not extra_records:
            raise RuntimeError("no extra training records selected by criteria")
        for rec in extra_records:
            v = ctx.spec.target.value(rec)
            if v is None or not rec.text:
                continue
            mode = ctx.spec.target.mode
            if str(mode.value) == "cardinal":
                try:
                    nv = float(v)
                except Exception:
                    continue
            else:
                s = str(v).strip()
                if not s:
                    continue
                nv = s
            train_texts_override.append(rec.text)
            train_labels_override.append(nv)
        ctx = UtilityCtx(ctx.dataset, ctx.data_in, ctx.annotations, ctx.spec, ctx.selection_criteria, ctx.debug, ctx.test_size, ctx.random_state, ctx.dry_run, ctx.output_format, ctx.output_file, ctx.identifier)
    experiment = TextUtilityExperiment(test_size=ctx.test_size, random_state=ctx.random_state)
    experiment.setup(
        target=ctx.spec.target,
        records=records,
        vectorizer=vectorizer,
        model=model,
        train_texts_override=train_texts_override or None,
        train_labels_override=train_labels_override or None,
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
                print(f"\n=== Training Dataset (Synthetic Override) ===")
                print(f"Predictions distribution: {dict(dist_preds_train)}")
                print(f"True distribution: {dict(dist_true_train)}")
                try:
                    all_labels_set = sorted(list(set(train_labels_override)))
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
                print(f"\n=== Validation Dataset (Original Records) ===")
                print(f"Predictions distribution: {dict(dist_preds)}")
                print(f"True distribution: {dict(dist_true)}")
                try:
                    all_labels_set = sorted(list(set(lbls_test)))
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
            print(f"\n=== Overall Dataset (Full Original Records) ===")
            print(f"Predictions distribution: {dict(dist_preds_all)}")
            print(f"True distribution: {dict(dist_true_all)}")
            try:
                cm_all = confusion_matrix(lbls_all, preds_all, labels=sorted(list(set(lbls_all))))
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
    records = load_records(ctx.dataset, ctx.data_in, ctx.max_records)
    if not records:
        raise RuntimeError("no records loaded")
    sources = collect_jsonl_sources(*ctx.annotations)
    if not sources:
        raise RuntimeError("no anonymized output files discovered")
    evaluation_inputs = build_divergence_evaluation_inputs(records, sources)
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
    records = load_records(ctx.dataset, ctx.data_in, ctx.max_records)
    if not records:
        raise RuntimeError("no records loaded")
    sources = collect_jsonl_sources(*ctx.annotations)
    if not sources:
        raise RuntimeError("no annotation files discovered")
    evaluation_datasets: Dict[str, List[DatasetRecord]] = {}
    evaluation_counts: Dict[str, int] = {}
    for name, path in sources.items():
        texts = read_texts_from_jsonl(path)
        dataset_records = build_privacy_evaluation_dataset_from_texts(records, texts)
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
