"""Regenerate indices/rat_bench/{train,val,test}.txt as a 60/20/20 split.

Identity clusters (same logic as RatBenchDatasetAdapter._assign_names) are kept
intact within a single split so no individual's records leak across splits.
Within that constraint, records are additionally stratified by `difficulty` so
train/val/test each get a representative 1:1:1 mix of difficulty 1/2/3 instead
of the previous split, which put ~100% of difficulty-3 records into val/test.

Cluster shape (fixed by the dataset, not chosen by this script):
  - 98 clusters of size 2, each exactly {difficulty 1, difficulty 2}
  - 99 clusters of size 1, difficulty 3
  - 2 leftover singleton clusters (1x difficulty 1, 1x difficulty 2)
  - 1 cluster of size 3, difficulty {1, 2, 3}
"""
from __future__ import annotations

import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from datasets import load_from_disk
from dp.loaders._ratbench import RatBenchDatasetAdapter, _parse_py_literal

SEED = 42
RATIOS = {"train": 0.6, "val": 0.2, "test": 0.2}


def load_rows() -> list[dict]:
    ds = load_from_disk("data/rat_bench")
    if hasattr(ds, "keys"):
        ds = ds["train"] if "train" in ds else next(iter(ds.values()))
    rows = []
    for idx in range(len(ds)):
        row = dict(ds[idx])
        row["profile"] = _parse_py_literal(row.get("profile") or {})
        row["direct_identifiers"] = _parse_py_literal(row.get("direct_identifiers") or {})
        row["indirect_identifiers"] = _parse_py_literal(row.get("indirect_identifiers") or {})
        rows.append(row)
    return rows


def identity_clusters(rows: list[dict]) -> list[list[int]]:
    rows_by_id: dict = {}
    for idx, row in enumerate(rows):
        rows_by_id.setdefault(row.get("id"), []).append(idx)
    clusters = []
    for record_id, indices in rows_by_id.items():
        clusters.extend(RatBenchDatasetAdapter._cluster_by_identity(rows, indices))
    return clusters


def split_counts(n: int) -> dict[str, int]:
    train = round(n * RATIOS["train"])
    val = round(n * RATIOS["val"])
    test = n - train - val
    return {"train": train, "val": val, "test": test}


def allocate(clusters: list[list[int]], rng: random.Random) -> dict[str, list[int]]:
    clusters = clusters[:]
    rng.shuffle(clusters)
    counts = split_counts(len(clusters))
    splits: dict[str, list[int]] = {"train": [], "val": [], "test": []}
    cursor = 0
    for name in ("train", "val", "test"):
        n = counts[name]
        for cluster in clusters[cursor : cursor + n]:
            splits[name].extend(cluster)
        cursor += n
    return splits


def main() -> None:
    rows = load_rows()
    clusters = identity_clusters(rows)

    pair_12 = [c for c in clusters if len(c) == 2]
    diff3_singletons = [c for c in clusters if len(c) == 1 and rows[c[0]]["difficulty"] == 3]
    leftover = [c for c in clusters if c not in pair_12 and c not in diff3_singletons]

    assert sum(len(c) for c in pair_12) + sum(len(c) for c in diff3_singletons) + sum(
        len(c) for c in leftover
    ) == len(rows)

    rng = random.Random(SEED)
    pair_splits = allocate(pair_12, rng)
    diff3_splits = allocate(diff3_singletons, rng)

    result = {"train": [], "val": [], "test": []}
    for name in result:
        result[name].extend(pair_splits[name])
        result[name].extend(diff3_splits[name])

    # Leftover clusters (2 stray singletons + 1 triple) are too few to stratify
    # meaningfully; assign deterministically to keep the split reproducible.
    leftover_target = ["train", "val", "test"]
    for i, cluster in enumerate(leftover):
        result[leftover_target[i % 3]].extend(cluster)

    for name in result:
        result[name].sort()

    out_dir = Path("indices/rat_bench")
    for name, indices in result.items():
        (out_dir / f"{name}.txt").write_text("\n".join(str(i) for i in indices) + "\n")

    total = sum(len(v) for v in result.values())
    print(f"seed={SEED}  total_records={total}")
    for name in ("train", "val", "test"):
        indices = result[name]
        diff_dist = Counter(rows[i]["difficulty"] for i in indices)
        scen_dist = Counter(rows[i]["scenario"] for i in indices)
        print(
            f"{name}: n={len(indices)} ({len(indices) / total:.1%})"
            f"  difficulty={dict(sorted(diff_dist.items()))}"
            f"  scenario={dict(scen_dist)}"
        )

    all_indices = set()
    for indices in result.values():
        overlap = all_indices & set(indices)
        assert not overlap, f"overlap detected: {overlap}"
        all_indices |= set(indices)
    assert all_indices == set(range(len(rows)))
    print("OK: no overlap, full coverage, no identity cluster split across sets.")


if __name__ == "__main__":
    main()
