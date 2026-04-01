"""
Compares token-selection trajectories across multiple T values pairwise.

For each record, one independent masking trajectory is run per T value:
  T=1   — SHAP re-computed every step (fully dynamic)
  T=k   — SHAP re-computed every k steps
  T=inf — precomputed scores used as-is, never refreshed

Trajectories diverge because each picks tokens according to its own score ranking,
which drifts after each refresh (or lack thereof).

At each step, for all (T_a, T_b) pairs:
  wasserstein  — EMD between n-length score vectors (0 for already-masked positions)
  jaccard      — set overlap between tokens accumulated so far by each trajectory

GPU memory is cleared after each SHAP recomputation.

Output: JSONL, one line per record.
"""

import argparse
import json
import math
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path
from typing import Any, Optional

import numpy as np
import yaml
from scipy.stats import wasserstein_distance
from tqdm import tqdm

from dp.loaders import get_adapter
from dp.utils.explainer import ShapExplainer, ShapType
from dp.utils.memory import clear_memory


@dataclass
class Trajectory:
    t: float
    remaining_orig: list[int]
    current_text: str
    current_offsets: list[tuple[int, int]]
    scores: np.ndarray
    accumulated: set[int] = field(default_factory=set)

    def padded_scores(self, n: int) -> np.ndarray:
        v = np.zeros(n, dtype=float)
        for local_i, orig_i in enumerate(self.remaining_orig):
            v[orig_i] = float(self.scores[local_i])
        return v


def _load_shap_jsonl(path: Path) -> dict[str, dict[str, Any]]:
    entries: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            entry = json.loads(line)
            entries[str(entry["uid"])] = entry
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
        adjusted.append((os + delta, oe + delta) if os >= e else (os, oe))
    return new_text, adjusted


def _pair_key(t_a: float, t_b: float) -> str:
    def _fmt(t: float) -> str:
        return "inf" if math.isinf(t) else str(int(t))
    return f"{_fmt(t_a)}_vs_{_fmt(t_b)}"


def _assess_record(
    text: str,
    offsets: list[tuple[int, int]],
    static_scores: np.ndarray,
    t_values: list[float],
    explainer: Optional[ShapExplainer],
    target_label: str,
    max_steps: Optional[int],
    mask_token: str,
) -> list[dict[str, Any]]:
    n = len(offsets)
    steps_to_run = n if max_steps is None else min(n, max_steps)
    t_pairs = list(combinations(t_values, 2))

    trajectories: dict[float, Trajectory] = {
        t: Trajectory(
            t=t,
            remaining_orig=list(range(n)),
            current_text=text,
            current_offsets=list(offsets),
            scores=static_scores.copy(),
        )
        for t in t_values
    }

    step_records: list[dict[str, Any]] = []

    for step in range(steps_to_run):
        # refresh scores for trajectories that need it
        initial_live_scores: Optional[np.ndarray] = None
        for t, traj in trajectories.items():
            if not traj.remaining_orig:
                continue
            if math.isinf(t):
                traj.scores = static_scores[traj.remaining_orig]
            elif step % int(t) == 0:
                if explainer is None:
                    raise ValueError("explainer_in required for finite T")
                # at step 0 all finite-T trajectories share the same initial text;
                # reuse the first live computation to avoid redundant GPU calls
                if step == 0 and initial_live_scores is not None:
                    traj.scores = initial_live_scores.copy()
                else:
                    raw = explainer.explain(
                        traj.current_text, traj.current_offsets, target_label=target_label
                    )
                    traj.scores = np.asarray(raw, dtype=float)
                    clear_memory()
                    if step == 0:
                        initial_live_scores = traj.scores.copy()

        # pairwise metrics before advancing
        padded = {t: traj.padded_scores(n) for t, traj in trajectories.items()}
        pair_metrics: dict[str, dict[str, float]] = {}
        for t_a, t_b in t_pairs:
            pair_metrics[_pair_key(t_a, t_b)] = {
                "wasserstein": float(wasserstein_distance(padded[t_a], padded[t_b])),
                "jaccard": _jaccard(trajectories[t_a].accumulated, trajectories[t_b].accumulated),
            }
        step_records.append({"step": step, "pairs": pair_metrics})

        # advance each trajectory independently
        for t, traj in trajectories.items():
            if not traj.remaining_orig:
                continue
            local_best = int(np.argmax(traj.scores))
            orig_best = traj.remaining_orig[local_best]
            traj.accumulated.add(orig_best)
            traj.current_text, traj.current_offsets = _mask_and_adjust(
                traj.current_text, traj.current_offsets, local_best, mask_token
            )
            traj.remaining_orig.pop(local_best)
            traj.scores = np.delete(traj.scores, local_best)

    return step_records


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
    explainer_type = ShapType(cfg.get("explainer_type", "shap"))
    explainer_in: str | None = cfg.get("explainer_in")
    shap_in = Path(cfg["shap_in"])
    output_dir = Path(cfg["output_dir"])
    mask_token: str = cfg.get("mask_token", "[MASK]")
    max_records: int | None = args.max_records or cfg.get("max_records")
    max_steps: int | None = args.max_steps or cfg.get("max_steps") or None

    raw_t_values: list[int | None] = cfg.get("t_values", [1, None])
    t_values: list[float] = [math.inf if v is None else float(v) for v in raw_t_values]

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "t_drift.jsonl"
    output_path.write_text("", encoding="utf-8")

    shap_data = _load_shap_jsonl(shap_in)

    adapter = get_adapter(dataset, data=dataset, data_in=data_in, split=split)
    records = list(adapter.iter_records())
    if max_records is not None:
        records = records[:max_records]

    needs_live_shap = any(not math.isinf(t) for t in t_values)
    explainer: Optional[ShapExplainer] = None
    label_mapping: dict[str, int] = {}
    if needs_live_shap:
        if explainer_in is None:
            raise ValueError("explainer_in required when any T value is finite")
        explainer = ShapExplainer(model_name=explainer_in, explainer_type=explainer_type)
        explainer._load_pipeline()
        label_mapping = explainer.label_to_id

    t_labels = ["inf" if math.isinf(t) else str(int(t)) for t in t_values]

    for record in tqdm(records, desc="T-drift"):
        uid_str = str(record.uid)
        if uid_str not in shap_data:
            continue

        entry = shap_data[uid_str]
        raw_offsets: list[tuple[int, int]] = [tuple(o) for o in entry["offsets"]]
        positional_order = sorted(range(len(raw_offsets)), key=lambda i: raw_offsets[i][0])
        offsets = [raw_offsets[i] for i in positional_order]
        static_scores = np.array(entry["scores"], dtype=float)[positional_order]

        if needs_live_shap:
            target_label_id = label_mapping.get(record.name)
            if target_label_id is None:
                continue
            target_label = f"LABEL_{target_label_id}"
        else:
            target_label = ""

        steps = _assess_record(
            text=record.text,
            offsets=offsets,
            static_scores=static_scores,
            t_values=t_values,
            explainer=explainer,
            target_label=target_label,
            max_steps=max_steps,
            mask_token=mask_token,
        )

        with output_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps({
                "uid": uid_str,
                "n_tokens": len(offsets),
                "t_values": t_labels,
                "steps": steps,
            }) + "\n")


if __name__ == "__main__":
    main()
