#!/usr/bin/env python3

from __future__ import annotations

import argparse
from typing import List, Optional

import numpy as np

from dp.loaders import ADAPTER_REGISTRY, DatasetRecord, get_adapter
from dp.utils.explainer import GreedyExplainer, ShapExplainer
from dp.utils.splitter import TextSplitter


def collect_records(
    dataset: str,
    data_in: str,
    max_records: Optional[int],
) -> List[DatasetRecord]:
    adapter = get_adapter(dataset, data=dataset, data_in=data_in, max_records=max_records)
    return list(adapter.iter_records())


def fetch_text(
    dataset: Optional[str],
    data_in: Optional[str],
    max_records: Optional[int],
    index: Optional[int],
    text: Optional[str],
) -> tuple[str, Optional[str]]:
    if text:
        return text, None
    if dataset is None or data_in is None:
        raise ValueError("dataset and data_in are required when text is not provided")
    records = collect_records(dataset, data_in, max_records)
    if not records:
        raise ValueError("dataset produced no records")
    resolved_index = 0 if index is None else index
    if resolved_index < 0 or resolved_index >= len(records):
        raise ValueError(f"index {resolved_index} is out of bounds for dataset of size {len(records)}")
    record = records[resolved_index]
    return record.text, record.name or None


def tokenize(text: str) -> List[str]:
    splitter = TextSplitter()
    return [token for _, _, token in splitter.tokenize_with_spans(text)]


def format_value(value: Optional[float]) -> str:
    if value is None:
        return "n/a"
    return f"{value:.6f}"

def to_distribution(values: Optional[np.ndarray], temperature: float) -> Optional[np.ndarray]:
    if values is None:
        return None
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    array = np.asarray(values, dtype=float)
    if array.size == 0:
        return None
    if not np.any(np.isfinite(array)):
        return None
    scaled = array / temperature
    scaled = scaled - np.max(scaled)
    exps = np.exp(scaled)
    total = np.sum(exps)
    if not np.isfinite(total) or total <= 0:
        length = array.shape[0]
        if length == 0:
            return None
        return np.full(length, 1.0 / length)
    return exps / total

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect Greedy and SHAP explanation scores")
    parser.add_argument("--model", required=True, help="HuggingFace model checkpoint for explainers")
    parser.add_argument("--dataset", choices=sorted(ADAPTER_REGISTRY.keys()), help="Dataset adapter name")
    parser.add_argument("--data-in", help="Path to dataset input")
    parser.add_argument("--max-records", type=int, help="Maximum records to load from dataset")
    parser.add_argument("--index", type=int, help="Record index to inspect (default: 0)")
    parser.add_argument("--text", help="Custom text to explain")
    parser.add_argument("--target-label", help="Target label override for explainers")
    parser.add_argument("--device", default="auto", help="Device for explainers (auto, cpu, cuda, mps)")
    parser.add_argument("--mask-token", default="[MASK]", help="Mask token used by greedy explainer")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size for greedy explainer")
    parser.add_argument("--use-chunking", action="store_true", help="Enable TRI chunking in explainers")
    parser.add_argument("--no-greedy", action="store_true", help="Skip greedy explainer")
    parser.add_argument("--no-shap", action="store_true", help="Skip SHAP explainer")
    parser.add_argument("--temperature", type=float, default=1.0, help="Temperature for probability conversion")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    text, record_label = fetch_text(
        dataset=args.dataset,
        data_in=args.data_in,
        max_records=args.max_records,
        index=args.index,
        text=args.text,
    )
    tokens = tokenize(text)
    if not tokens:
        raise ValueError("tokenization produced no tokens")
    target_label = args.target_label
    greedy_scores: Optional[np.ndarray] = None
    shap_scores: Optional[np.ndarray] = None
    greedy_probs: Optional[np.ndarray] = None
    shap_probs: Optional[np.ndarray] = None
    if not args.no_greedy:
        greedy = GreedyExplainer(
            model_name=args.model,
            mask_token=args.mask_token,
            batch_size=args.batch_size,
            device=args.device,
            use_chunking=args.use_chunking,
        )
        greedy_scores = greedy.explain(text, tokens, target_label=target_label)
        greedy_probs = to_distribution(greedy_scores, args.temperature)
    if not args.no_shap:
        shap = ShapExplainer(
            model_name=args.model,
            device=args.device,
            use_chunking=args.use_chunking,
        )
        shap_scores = shap.explain(text, tokens, target_label=target_label)
        shap_probs = to_distribution(shap_scores, args.temperature)
    print("Text:")
    print(text)
    if record_label:
        print(f"Record label: {record_label}")
    if target_label:
        print(f"Target label: {target_label}")
    print("")
    header_parts = ["idx", "token"]
    if greedy_scores is not None:
        header_parts.append("greedy")
    if shap_scores is not None:
        header_parts.append("shap")
    if greedy_probs is not None:
        header_parts.append("greedy_prob")
    if shap_probs is not None:
        header_parts.append("shap_prob")
    print("\t\t".join(header_parts))
    for idx, token in enumerate(tokens):
        row = [str(idx), token]
        if greedy_scores is not None:
            value = greedy_scores[idx] if idx < len(greedy_scores) else None
            row.append(format_value(value))
        if shap_scores is not None:
            value = shap_scores[idx] if idx < len(shap_scores) else None
            row.append(format_value(value))
        if greedy_probs is not None:
            value = greedy_probs[idx] if idx < len(greedy_probs) else None
            row.append(format_value(value))
        if shap_probs is not None:
            value = shap_probs[idx] if idx < len(shap_probs) else None
            row.append(format_value(value))
        print("\t\t".join(row))


if __name__ == "__main__":
    main()
