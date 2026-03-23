import json
from pathlib import Path
from typing import Literal

import numpy as np
from scipy.stats import wasserstein_distance


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

    per_record = per_record_distances(full, exact, common_uids)
    print(f"Per-record Wasserstein distances:")
    print(f"  mean: {per_record.mean():.6f}, std: {per_record.std():.6f}, "
          f"min: {per_record.min():.6f}, max: {per_record.max():.6f}")

    top_k = 10
    top_indices = np.argsort(per_record)[::-1][:top_k]
    print(f"  Top {top_k} UIDs by distance:")
    for i in top_indices:
        print(f"    {common_uids[i]}: {per_record[i]:.6f}")


if __name__ == "__main__":
    for dataset in DATASETS:
        analyze(dataset)
