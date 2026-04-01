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

All trajectories operate on the de-identified text (result_in), consistent with
how risk scores were precomputed. SHAP refreshes are run on the partially masked
de-identified text, matching the domain the classifier was trained on.

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
from dp.loaders.results import build_dataset_from_results, load_result_records
from dp.utils.explainer import ShapExplainer, ShapType
from dp.utils.memory import clear_memory
from dp.utils.splitter import TextSplitter
from dp.utils.token_ledger import TokenLedger


@dataclass
class Trajectory:
    t: float
    ledger: TokenLedger
    remaining_orig: list[int]
    scores: np.ndarray
    accumulated: set[int] = field(default_factory=set)

    def padded_scores(self) -> np.ndarray:
        v = np.zeros(len(self.scores), dtype=float)
        for i in self.remaining_orig:
            v[i] = float(self.scores[i])
        return v


def _compute_current_offsets(ledger: TokenLedger, n: int) -> list[tuple[int, int]]:
    result: list[Optional[tuple[int, int]]] = [None] * n
    sorted_entries = sorted(ledger._entries, key=lambda e: e.start)
    cursor = 0
    gap_idx = 0
    gaps = ledger._gaps
    for entry in sorted_entries:
        while gap_idx < len(gaps) and gaps[gap_idx].start < entry.start:
            cursor += len(gaps[gap_idx].text)
            gap_idx += 1
        if not entry.deleted:
            result[entry.index] = (cursor, cursor + len(entry.text))
            cursor += len(entry.text)
    for i in range(n):
        if result[i] is None:
            result[i] = (0, 0)
    return result  # type: ignore


def _refresh_scores(
    ledger: TokenLedger,
    deid_text: str,
    n: int,
    remaining: list[int],
    explainer: ShapExplainer,
    target_label: str,
) -> np.ndarray:
    current_text = ledger.render_offsets(deid_text)
    current_offsets = _compute_current_offsets(ledger, n)
    surviving = [(i, current_offsets[i]) for i in range(n) if not ledger.entry(i).deleted]
    if not surviving:
        return np.zeros(n, dtype=float)
    surv_idx, surv_offs = zip(*surviving)
    raw = explainer.explain(current_text, list(surv_offs), target_label=target_label)
    full_scores = np.zeros(n, dtype=float)
    for pos, orig_i in enumerate(surv_idx):
        full_scores[orig_i] = float(raw[pos])
    remaining_set = set(remaining)
    for i in range(n):
        if i not in remaining_set:
            full_scores[i] = 0.0
    return full_scores


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


def _pair_key(t_a: float, t_b: float) -> str:
    def _fmt(t: float) -> str:
        return "inf" if math.isinf(t) else str(int(t))
    return f"{_fmt(t_a)}_vs_{_fmt(t_b)}"


def _assess_record(
    deid_text: str,
    deid_offsets: list[tuple[int, int]],
    static_scores: np.ndarray,
    t_values: list[float],
    explainer: Optional[ShapExplainer],
    target_label: str,
    max_steps: Optional[int],
    mask_token: str,
) -> list[dict[str, Any]]:
    n = len(deid_offsets)
    steps_to_run = n if max_steps is None else min(n, max_steps)
    t_pairs = list(combinations(t_values, 2))

    trajectories: dict[float, Trajectory] = {
        t: Trajectory(
            t=t,
            ledger=TokenLedger(deid_text, deid_offsets),
            remaining_orig=list(range(n)),
            scores=static_scores.copy(),
        )
        for t in t_values
    }

    step_records: list[dict[str, Any]] = []
    initial_live_scores: Optional[np.ndarray] = None

    for step in range(steps_to_run):
        for t, traj in trajectories.items():
            if not traj.remaining_orig:
                continue
            if math.isinf(t):
                scores = static_scores.copy()
                remaining_set = set(traj.remaining_orig)
                for i in range(n):
                    if i not in remaining_set:
                        scores[i] = 0.0
                traj.scores = scores
            elif step % int(t) == 0:
                if explainer is None:
                    raise ValueError("explainer_in required for finite T")
                if step == 0 and initial_live_scores is not None:
                    traj.scores = initial_live_scores.copy()
                else:
                    traj.scores = _refresh_scores(
                        traj.ledger, deid_text, n, traj.remaining_orig,
                        explainer, target_label,
                    )
                    clear_memory()
                    if step == 0:
                        initial_live_scores = traj.scores.copy()

        pair_metrics: dict[str, dict[str, float]] = {}
        for t_a, t_b in t_pairs:
            pair_metrics[_pair_key(t_a, t_b)] = {
                "wasserstein": float(wasserstein_distance(
                    trajectories[t_a].padded_scores(),
                    trajectories[t_b].padded_scores(),
                )),
                "jaccard": _jaccard(trajectories[t_a].accumulated, trajectories[t_b].accumulated),
            }
        step_records.append({"step": step, "pairs": pair_metrics})

        for t, traj in trajectories.items():
            if not traj.remaining_orig:
                continue
            orig_best = max(traj.remaining_orig, key=lambda i: float(traj.scores[i]))
            traj.accumulated.add(orig_best)
            traj.ledger.replace(orig_best, mask_token)
            traj.remaining_orig.remove(orig_best)
            traj.scores[orig_best] = 0.0

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
    result_in: str = cfg["result_in"]
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

    splitter = TextSplitter()

    all_original = list(get_adapter(dataset, data=dataset, data_in=data_in).iter_records())
    deid_records, _ = build_dataset_from_results(result_in, all_original)
    deid_by_uid: dict[str, str] = {str(r.uid): r.text for r in deid_records}

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
        if uid_str not in deid_by_uid:
            continue

        deid_text = deid_by_uid[uid_str]
        deid_offsets = [(s, e) for s, e, _ in splitter.tokenize_with_spans(deid_text)]

        entry = shap_data[uid_str]
        raw_offsets: list[tuple[int, int]] = [tuple(o) for o in entry["offsets"]]
        if len(raw_offsets) != len(deid_offsets):
            continue

        positional_order = sorted(range(len(raw_offsets)), key=lambda i: raw_offsets[i][0])
        static_scores = np.array(entry["scores"], dtype=float)[positional_order]

        if needs_live_shap:
            target_label_id = label_mapping.get(record.name)
            if target_label_id is None:
                continue
            target_label = f"LABEL_{target_label_id}"
        else:
            target_label = ""

        steps = _assess_record(
            deid_text=deid_text,
            deid_offsets=deid_offsets,
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
                "n_tokens": len(deid_offsets),
                "t_values": t_labels,
                "steps": steps,
            }) + "\n")


if __name__ == "__main__":
    main()
