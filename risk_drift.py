"""
Measures how much SHAP scores drift as tokens are iteratively masked.

For each record, two orderings are compared:
  static  — tokens ranked once by precomputed SHAP scores, never updated
  dynamic — after each masking step, SHAP is re-computed on the modified text
             and the next token is chosen from the updated ranking

At every step the following are recorded:
  wasserstein  — Earth Mover's Distance between static and dynamic scores
                 over the set of tokens not yet handled
  jaccard      — overlap between the accumulated static and dynamic token sets
"""

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from scipy.stats import wasserstein_distance
from tqdm import tqdm

from dp.loaders import get_adapter
from dp.utils.explainer import ShapExplainer, ShapType
from dp.utils.memory import clear_memory


@dataclass
class StepRecord:
    step: int
    n_remaining: int
    wasserstein: float
    jaccard: float
    static_orig_idx: int
    dynamic_orig_idx: int


def _load_shap_jsonl(path: Path) -> dict[str, dict[str, Any]]:
    entries: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            entry = json.loads(line)
            entries[entry["uid"]] = entry
    return entries


def _jaccard(a: set[int], b: set[int]) -> float:
    union = a | b
    return len(a & b) / len(union) if union else 1.0


def _mask_and_adjust(
    text: str,
    offsets: list[tuple[int, int]],
    local_idx: int,
    mask_token: str,
) -> tuple[str, list[tuple[int, int]]]:
    s, e = offsets[local_idx]
    new_text = text[:s] + mask_token + text[e:]
    delta = len(mask_token) - (e - s)
    adjusted: list[tuple[int, int]] = []
    for i, (os, oe) in enumerate(offsets):
        if i == local_idx:
            continue
        if os >= e:
            adjusted.append((os + delta, oe + delta))
        else:
            adjusted.append((os, oe))
    return new_text, adjusted


def _assess_record(
    uid: str,
    text: str,
    offsets: list[tuple[int, int]],
    static_scores: np.ndarray,
    explainer: ShapExplainer,
    target_label: str,
    max_steps: int,
    mask_token: str,
) -> list[StepRecord]:
    n = len(offsets)
    steps_to_run = n if max_steps is None else min(n, max_steps)

    static_order = list(np.argsort(static_scores)[::-1])

    current_text = text
    current_offsets = list(offsets)
    remaining_orig: list[int] = list(range(n))
    dyn_scores = static_scores.copy()

    static_set: set[int] = set()
    dynamic_set: set[int] = set()
    results: list[StepRecord] = []

    for step in range(steps_to_run):
        static_orig = static_order[step]
        static_set.add(static_orig)

        local_dyn = int(np.argmax(dyn_scores))
        dynamic_orig = remaining_orig[local_dyn]
        dynamic_set.add(dynamic_orig)

        remaining_static = static_scores[remaining_orig]
        wd = float(wasserstein_distance(remaining_static, dyn_scores))
        j = _jaccard(static_set, dynamic_set)

        results.append(StepRecord(
            step=step,
            n_remaining=len(remaining_orig),
            wasserstein=wd,
            jaccard=j,
            static_orig_idx=static_orig,
            dynamic_orig_idx=dynamic_orig,
        ))

        current_text, current_offsets = _mask_and_adjust(
            current_text, current_offsets, local_dyn, mask_token
        )
        remaining_orig.pop(local_dyn)

        if not remaining_orig:
            break

        try:
            dyn_scores = explainer.explain(
                current_text, current_offsets, target_label=target_label
            )
            clear_memory()
        except Exception:
            dyn_scores = static_scores[remaining_orig]

    return results


def _load_config(path: str) -> dict[str, Any]:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--max_records", type=int, default=None)
    parser.add_argument("--max_steps", type=int, default=None)
    args = parser.parse_args()

    cfg = _load_config(args.config)
    dataset: str = cfg["dataset"]
    data_in: str = cfg["data_in"]
    split: str | None = cfg.get("split")
    explainer_type = ShapType(cfg.get("explainer", "shap"))
    explainer_in: str = cfg["explainer_in"]
    shap_in = Path(cfg["shap_in"])
    output_dir = Path(cfg["output_dir"])
    mask_token: str = cfg.get("mask_token", "[MASK]")
    max_records: int | None = args.max_records or cfg.get("max_records")
    max_steps: int | None = args.max_steps or cfg.get("max_steps") or None

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "drift.jsonl"
    output_path.write_text("", encoding="utf-8")

    shap_data = _load_shap_jsonl(shap_in)

    adapter = get_adapter(dataset, data=dataset, data_in=data_in, split=split)
    records = list(adapter.iter_records())
    if max_records is not None:
        records = records[:max_records]

    explainer = ShapExplainer(model_name=explainer_in, explainer_type=explainer_type)
    explainer._load_pipeline()
    label_mapping = explainer.label_to_id

    for record in tqdm(records, desc="Assessing drift"):
        if record.uid not in shap_data:
            continue

        entry = shap_data[record.uid]
        raw_offsets: list[tuple[int, int]] = [tuple(o) for o in entry["offsets"]]
        positional_order = sorted(range(len(raw_offsets)), key=lambda i: raw_offsets[i][0])
        offsets = [raw_offsets[i] for i in positional_order]
        scores = np.array(entry["scores"])[positional_order]

        target_label_id = label_mapping.get(record.name)
        if target_label_id is None:
            continue

        steps = _assess_record(
            uid=record.uid,
            text=record.text,
            offsets=offsets,
            static_scores=scores,
            explainer=explainer,
            target_label=f"LABEL_{target_label_id}",
            max_steps=max_steps,
            mask_token=mask_token,
        )

        with output_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps({
                "uid": record.uid,
                "n_tokens": len(offsets),
                "steps": [asdict(s) for s in steps],
            }) + "\n")


if __name__ == "__main__":
    main()
