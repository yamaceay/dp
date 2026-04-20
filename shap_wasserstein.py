import json
from pathlib import Path
from typing import Literal

import numpy as np
from scipy.stats import spearmanr, wasserstein_distance

from dp.utils.risk import _scores_to_inverse_probs


Dataset = Literal["db_bio", "tab"]

DATASETS: list[Dataset] = ["db_bio", "tab"]

DATA_DIR = Path("data")


def load_shap(path: Path) -> dict[str, list[float]]:
    records: dict[str, list[float]] = {}
    with path.open() as f:
        for line in f:
            entry = json.loads(line)
            records[entry["uid"]] = entry["scores"]
    return records


def flatten_scores(shap: dict[str, list[float]], uids: list[str]) -> np.ndarray:
    return np.array([score for uid in uids for score in shap[uid]])


def per_record_distances(
    full: dict[str, list[float]],
    exact: dict[str, list[float]],
    uids: list[str],
) -> np.ndarray:
    return np.array(
        [
            wasserstein_distance(full[uid], exact[uid])
            for uid in uids
        ]
    )


def per_record_spearman(
    full: dict[str, list[float]],
    exact: dict[str, list[float]],
    uids: list[str],
) -> tuple[np.ndarray, int]:
    correlations: list[float] = []
    skipped = 0
    for uid in uids:
        full_scores = np.asarray(full[uid], dtype=float)
        exact_scores = np.asarray(exact[uid], dtype=float)
        if len(full_scores) != len(exact_scores) or len(full_scores) < 2:
            skipped += 1
            continue
        corr = spearmanr(full_scores, exact_scores).correlation
        if np.isfinite(corr):
            correlations.append(float(corr))
    return np.asarray(correlations, dtype=float), skipped


def allocation_dispersion(
    shap: dict[str, list[float]],
    uids: list[str],
    *,
    temperature: float = 0.1,
) -> dict[str, float]:
    stds: list[float] = []
    maxes: list[float] = []
    entropies: list[float] = []
    for uid in uids:
        scores = np.asarray(shap[uid], dtype=float)
        if len(scores) < 2:
            continue
        weights = _scores_to_inverse_probs(scores, temperature=temperature)
        stds.append(float(np.std(weights)))
        maxes.append(float(np.max(weights)))
        entropies.append(float(-(weights * np.log(weights + 1e-12)).sum()))

    return {
        "mean_std": float(np.mean(stds)) if stds else 0.0,
        "mean_max": float(np.mean(maxes)) if maxes else 0.0,
        "mean_entropy": float(np.mean(entropies)) if entropies else 0.0,
    }


def analyze(dataset: Dataset) -> None:
    full_path = DATA_DIR / dataset / "tri_risk" / "shap.jsonl"
    exact_path = DATA_DIR / dataset / "tri_risk" / "shap_subset.jsonl"

    full = load_shap(full_path)
    exact = load_shap(exact_path)

    common_uids = sorted(set(full) & set(exact))
    print(f"\n=== {dataset} ===")
    print(f"Full records: {len(full)}, Exact records: {len(exact)}, Common: {len(common_uids)}")

    full_flat = flatten_scores(full, common_uids)
    exact_flat = flatten_scores(exact, common_uids)

    global_dist = wasserstein_distance(full_flat, exact_flat)
    print(f"Global Wasserstein distance (pooled tokens): {global_dist:.6f}")

    print(f"  Full  — mean: {full_flat.mean():.6f}, std: {full_flat.std():.6f}, "
          f"min: {full_flat.min():.6f}, max: {full_flat.max():.6f}")
    print(f"  Exact — mean: {exact_flat.mean():.6f}, std: {exact_flat.std():.6f}, "
          f"min: {exact_flat.min():.6f}, max: {exact_flat.max():.6f}")
    full_var = float(np.var(full_flat))
    exact_var = float(np.var(exact_flat))
    print(
        f"  Relative variance (Full / Exact): {full_var / max(exact_var, 1e-12):.3f}x "
        f"(Full: {full_var:.8f}, Exact: {exact_var:.8f})"
    )

    per_record = per_record_distances(full, exact, common_uids)
    print(f"Per-record Wasserstein distances:")
    print(f"  mean: {per_record.mean():.6f}, std: {per_record.std():.6f}, "
          f"min: {per_record.min():.6f}, max: {per_record.max():.6f}")

    top_k = 10
    top_indices = np.argsort(per_record)[::-1][:top_k]
    print(f"  Top {top_k} UIDs by distance:")
    for i in top_indices:
        print(f"    {common_uids[i]}: {per_record[i]:.6f}")

    rank_corr, skipped = per_record_spearman(full, exact, common_uids)
    if rank_corr.size:
        print("Per-record rank preservation (Spearman correlation):")
        print(
            f"  n: {rank_corr.size} (skipped: {skipped}), mean: {rank_corr.mean():.6f}, "
            f"median: {np.median(rank_corr):.6f}, min: {rank_corr.min():.6f}"
        )

    full_alloc = allocation_dispersion(full, common_uids, temperature=0.1)
    exact_alloc = allocation_dispersion(exact, common_uids, temperature=0.1)
    print("Post-normalization allocation dispersion (negated-risk softmax, tau=0.1):")
    print(
        f"  Mean std(weights)  — Full: {full_alloc['mean_std']:.6f}, "
        f"Exact: {exact_alloc['mean_std']:.6f}, "
        f"ratio: {full_alloc['mean_std'] / max(exact_alloc['mean_std'], 1e-12):.3f}x"
    )
    print(
        f"  Mean max(weights)  — Full: {full_alloc['mean_max']:.6f}, "
        f"Exact: {exact_alloc['mean_max']:.6f}, "
        f"ratio: {full_alloc['mean_max'] / max(exact_alloc['mean_max'], 1e-12):.3f}x"
    )
    print(
        f"  Mean entropy       — Full: {full_alloc['mean_entropy']:.6f}, "
        f"Exact: {exact_alloc['mean_entropy']:.6f}"
    )


if __name__ == "__main__":
    for dataset in DATASETS:
        analyze(dataset)
