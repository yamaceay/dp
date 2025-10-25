#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import yaml

from dp.experiments.divergence.io import (
    build_divergence_evaluation_inputs,
    build_original_texts,
    build_record_info,
)
from dp.experiments.divergence.reporting import build_divergence_report, create_divergence_outputter
from dp.experiments.privacy.io import build_privacy_evaluation_dataset
from dp.experiments.privacy.reporting import build_privacy_report, create_privacy_outputter
from dp.experiments.utility.io import build_utility_evaluation_texts
from dp.experiments.utility.reporting import build_utility_report, create_utility_outputter
from dp.experiments.utility.models import (
    UTILITY_EXPERIMENTS_REGISTRY,
    UtilitySpec,
)
from dp.experiments.divergence.bertscore import BERTScoreDivergence
from dp.experiments.divergence.cosine import CosineSimilarityDivergence
from dp.experiments.divergence.base import TextDivergenceExperiment
from dp.experiments.privacy_annotations import TextPrivacyExperiment
from dp.experiments.utility.base import TextUtilityExperiment
from dp.experiments.utility.vectorizer import TfidfTextVectorizer, BERTVectorizer, SelfSupervisedFeatureExtractor
from dp.experiments.utils import build_output_sink, collect_jsonl_sources
from dp.loaders import DatasetRecord, get_adapter
from dp.loaders.annotations import read_batch_annotations_from_path


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG_DIR = PROJECT_ROOT / "configs" / "experiments"


def load_records(dataset: str, data_in: Optional[str], max_records: Optional[int]) -> List[DatasetRecord]:
    adapter = get_adapter(dataset, data=dataset, data_in=data_in, max_records=max_records)
    return list(adapter.iter_records())


def resolve_config_path(value: str) -> Path:
    raw = Path(value).expanduser()
    seen: Set[Path] = set()
    candidates: List[Path] = []

    def register(path: Path) -> None:
        if path in seen:
            return
        candidates.append(path)
        seen.add(path)

    def register_with_suffixes(path: Path) -> None:
        if path.suffix:
            register(path)
        else:
            register(path.with_suffix(".yaml"))
            register(path.with_suffix(".yml"))

    register_with_suffixes(raw)

    for candidate in list(candidates):
        if not candidate.is_absolute():
            register((PROJECT_ROOT / candidate).expanduser())

    if raw.suffix:
        register(DEFAULT_CONFIG_DIR / raw.name)
    else:
        register(DEFAULT_CONFIG_DIR / f"{raw.name}.yaml")
        register(DEFAULT_CONFIG_DIR / f"{raw.name}.yml")

    for candidate in candidates:
        if candidate.exists():
            return candidate
    names = ", ".join(str(c) for c in candidates)
    raise FileNotFoundError(f"config not found ({names})")


def load_config(value: Optional[str]) -> Dict[str, Any]:
    if value is None:
        return {}
    path = resolve_config_path(value)
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"config must be a mapping ({path})")
    return data


def merge_params(config: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    merged = dict(config)
    arg_map = vars(args)
    for key in ("mode", "identifier"):
        if key in arg_map and arg_map[key] is not None:
            merged[key] = arg_map[key]
    return merged


def ensure_sequence(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return [str(item) for item in value]
    return [str(value)]


def normalize_output_settings(params: Dict[str, Any]) -> None:
    output = params.get("output")
    if isinstance(output, dict):
        if "format" in output and "output_format" not in params:
            params["output_format"] = output["format"]
        if "file" in output and "output_file" not in params:
            params["output_file"] = output["file"]


def parse_component_config(payload: Any) -> Tuple[str, Dict[str, Any]]:
    if payload is None:
        return "", {}
    if isinstance(payload, str):
        return payload, {}
    if isinstance(payload, dict):
        cfg = dict(payload)
        name = str(cfg.pop("type", cfg.pop("name", "")))
        params = cfg.get("params", {}) if "params" in cfg else cfg
        if not isinstance(params, dict):
            raise ValueError("component params must be a mapping")
        return name, dict(params)
    raise ValueError("invalid component config")

def parse_metric_config(payload: Any) -> Tuple[str, Dict[str, Any]]:
    if payload is None:
        return "bertscore", {}
    if isinstance(payload, str):
        return payload, {}
    if isinstance(payload, dict):
        config = dict(payload)
        metric_type = str(config.pop("type", "bertscore"))
        return metric_type, config
    raise ValueError("metric config must be a string or mapping")


def build_vectorizer_from_config(payload: Any) -> SelfSupervisedFeatureExtractor:
    name, params = parse_component_config(payload)
    if not name:
        name = "tfidf"
    name = name.lower()
    if name == "tfidf":
        return TfidfTextVectorizer(**params)
    if name == "bert":
        return BERTVectorizer(**params)
    raise ValueError(f"unsupported vectorizer '{name}'")


def handle_utility(args: argparse.Namespace, config: Dict[str, Any]) -> None:
    params = merge_params(config, args)
    normalize_output_settings(params)
    annotation_values = ensure_sequence(params.get("annotations")) + ensure_sequence(params.get("annotations_in"))
    annotations = list(dict.fromkeys(annotation_values))
    dataset = params.get("dataset")
    data_in = params.get("data_in")
    target = params.get("target")
    if not dataset:
        raise ValueError("dataset is required")
    if not data_in:
        raise ValueError("data_in is required")
    if not annotations:
        raise ValueError("annotations are required")
    if not target:
        raise ValueError("target is required")
    max_records = params.get("max_records")
    test_size = params.get("test_size", 0.2)
    random_state = params.get("random_state", 42)
    dry_run = params.get("dry_run", False)
    output_format = params.get("output_format", "text")
    output_file = params.get("output_file")
    records = load_records(dataset, data_in, max_records)
    if not records:
        raise RuntimeError("no records loaded")
    spec_key = f"{dataset}_{target}"
    spec: Optional[UtilitySpec] = UTILITY_EXPERIMENTS_REGISTRY.get(spec_key)
    if spec is None:
        dataset_prefix = f"{dataset}_"
        available = sorted(
            key[len(dataset_prefix) :]
            for key in UTILITY_EXPERIMENTS_REGISTRY.keys()
            if key.startswith(dataset_prefix)
        )
        raise ValueError(f"unknown utility target '{target}' for dataset '{dataset}' (available: {', '.join(available)})")
    if dry_run:
        coverage = sum(1 for record in records if spec.target.value(record) is not None and record.text)
        print(f"Records loaded: {len(records)}")
        print(f"Target coverage: {coverage}")
        return
    sources = collect_jsonl_sources(*annotations)
    if not sources:
        raise RuntimeError("no anonymized output files discovered")
    index_to_key = {idx: record.uid for idx, record in enumerate(records)}
    evaluation_texts = build_utility_evaluation_texts(index_to_key, sources)
    evaluation_texts = {name: mapping for name, mapping in evaluation_texts.items() if mapping}
    if not evaluation_texts:
        raise RuntimeError("no anonymized texts aligned with dataset records")
    util_cfg = params.get("utility", {}) or {}
    if not util_cfg:
        legacy_vec = params.get("vectorizer")
        legacy_head = params.get("head")
        util_cfg = {"vectorizer": legacy_vec, "head": legacy_head}
    vec_name, vec_kwargs = parse_component_config(util_cfg.get("vectorizer"))
    head_name, head_kwargs = parse_component_config(util_cfg.get("head"))
    identifier = params.get("identifier") or util_cfg.get("identifier")
    vectorizer, model = spec.build_components(
        vectorizer_name=vec_name or None,
        vectorizer_kwargs=vec_kwargs,
        head_name=head_name or None,
        head_kwargs=head_kwargs,
        identifier=identifier,
    )
    experiment = TextUtilityExperiment(test_size=float(test_size), random_state=int(random_state))
    experiment.setup(target=spec.target, records=records, vectorizer=vectorizer, model=model)
    result = experiment.run(evaluation_texts=evaluation_texts)
    experiment.cleanup()
    report = build_utility_report(result, sources)
    sink = build_output_sink(output_file)
    outputter = create_utility_outputter(output_format, sink)
    outputter.output(report)


def build_divergence_experiment(metric_type: str, metric_params: Dict[str, Any]) -> TextDivergenceExperiment:
    if metric_type == "bertscore":
        allowed = {"model_type", "language", "batch_size", "device", "rescale_with_baseline"}
        kwargs = {key: metric_params[key] for key in allowed if key in metric_params}
        return BERTScoreDivergence(**kwargs)
    if metric_type == "cosine":
        vectorizer = build_vectorizer_from_config(metric_params.get("vectorizer"))
        return CosineSimilarityDivergence(vectorizer=vectorizer)
    raise ValueError(f"unsupported divergence metric '{metric_type}'")


def handle_divergence(args: argparse.Namespace, config: Dict[str, Any]) -> None:
    params = merge_params(config, args)
    normalize_output_settings(params)
    annotation_values = ensure_sequence(params.get("annotations")) + ensure_sequence(params.get("annotations_in"))
    annotations = list(dict.fromkeys(annotation_values))
    dataset = params.get("dataset")
    data_in = params.get("data_in")
    if not dataset:
        raise ValueError("dataset is required")
    if not data_in:
        raise ValueError("data_in is required")
    if not annotations:
        raise ValueError("annotations are required")
    max_records = params.get("max_records")
    output_format = params.get("output_format", "text")
    output_file = params.get("output_file")
    metric_type, metric_params = parse_metric_config(params.get("metric"))
    records = load_records(dataset, data_in, max_records)
    if not records:
        raise RuntimeError("no records loaded")
    sources = collect_jsonl_sources(*annotations)
    if not sources:
        raise RuntimeError("no anonymized output files discovered")
    evaluation_inputs = build_divergence_evaluation_inputs(records, sources)
    evaluation_inputs = {name: payload for name, payload in evaluation_inputs.items() if payload.get("texts")}
    if not evaluation_inputs:
        raise RuntimeError("no anonymized outputs aligned with dataset records")
    record_info = build_record_info(records)
    original_texts = build_original_texts(records)
    experiment = build_divergence_experiment(metric_type, metric_params)
    experiment.setup(
        original_texts=original_texts,
        evaluation_datasets=evaluation_inputs,
        record_info=record_info,
    )
    result = experiment.run()
    experiment.cleanup()
    report = build_divergence_report(result, len(records), sources)
    sink = build_output_sink(output_file)
    outputter = create_divergence_outputter(output_format, sink)
    outputter.output(report)


def handle_privacy(args: argparse.Namespace, config: Dict[str, Any]) -> None:
    params = merge_params(config, args)
    normalize_output_settings(params)
    tri_cfg = params.get("tri")
    if isinstance(tri_cfg, dict):
        if "pipeline" in tri_cfg and "tri_pipeline" not in params:
            params["tri_pipeline"] = tri_cfg["pipeline"]
        if "max_length" in tri_cfg and "tri_max_length" not in params:
            params["tri_max_length"] = tri_cfg["max_length"]
        if "device" in tri_cfg and "tri_device" not in params:
            params["tri_device"] = tri_cfg["device"]
    annotation_values = ensure_sequence(params.get("annotations")) + ensure_sequence(params.get("annotations_in"))
    annotations = list(dict.fromkeys(annotation_values))
    dataset = params.get("dataset")
    data_in = params.get("data_in")
    tri_pipeline = params.get("tri_pipeline")
    if not dataset:
        raise ValueError("dataset is required")
    if not data_in:
        raise ValueError("data_in is required")
    if not tri_pipeline:
        raise ValueError("tri_pipeline is required")
    if not annotations:
        raise ValueError("annotations are required")
    max_records = params.get("max_records")
    mask_token = params.get("mask_token", "[MASK]")
    tri_max_length = params.get("tri_max_length", 512)
    tri_device = params.get("tri_device", "auto")
    if params.get("no_progress"):
        progress = False
    elif "progress" in params:
        progress = bool(params.get("progress"))
    else:
        progress = True
    output_format = params.get("output_format", "text")
    output_file = params.get("output_file")
    records = load_records(dataset, data_in, max_records)
    if not records:
        raise RuntimeError("no records loaded")
    sources = collect_jsonl_sources(*annotations)
    if not sources:
        raise RuntimeError("no annotation files discovered")
    evaluation_datasets: Dict[str, List[DatasetRecord]] = {}
    evaluation_counts: Dict[str, int] = {}
    for name, path in sources.items():
        annotations_batch = read_batch_annotations_from_path(str(path))
        dataset_records = build_privacy_evaluation_dataset(records, annotations_batch, mask_token)
        evaluation_datasets[name] = dataset_records
        evaluation_counts[name] = len(dataset_records)
    experiment = TextPrivacyExperiment(
        tri_pipeline=tri_pipeline,
        tri_max_length=int(tri_max_length),
        tri_device=str(tri_device),
    )
    experiment.setup(
        dataset_name=dataset,
        original_dataset=records,
        evaluation_datasets=evaluation_datasets,
        progress=bool(progress),
    )
    result = experiment.run(progress=bool(progress))
    experiment.cleanup()
    report = build_privacy_report(
        result=result,
        original_record_count=len(records),
        annotation_sources=sources,
        evaluation_record_counts=evaluation_counts,
    )
    sink = build_output_sink(output_file)
    outputter = create_privacy_outputter(output_format, sink)
    outputter.output(report)


def build_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", type=str, help="Path to experiment config file")
    parser.add_argument("--mode", type=str, choices=["utility", "privacy", "divergence"], help="Experiment mode override (optional)")
    parser.add_argument("--identifier", type=str, help="Optional model identifier override for utility experiments")

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run experiments",
        argument_default=argparse.SUPPRESS,
    )
    subparsers = parser.add_subparsers(dest="command")
    subparsers.required = True

    utility = subparsers.add_parser("utility", help="Run utility experiment", argument_default=argparse.SUPPRESS)
    build_args(utility)
    utility.set_defaults(handler=handle_utility)

    divergence = subparsers.add_parser("divergence", help="Run divergence experiment", argument_default=argparse.SUPPRESS)
    build_args(divergence)
    divergence.set_defaults(handler=handle_divergence)

    privacy = subparsers.add_parser("privacy", help="Run privacy experiment", argument_default=argparse.SUPPRESS)
    build_args(privacy)
    privacy.set_defaults(handler=handle_privacy)

    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    handler = getattr(args, "handler")
    config_path = getattr(args, "config", None)
    config = load_config(config_path)
    try:
        handler(args, config)
    except Exception as exc:
        raise SystemExit(str(exc))


if __name__ == "__main__":
    main()
