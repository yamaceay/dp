import argparse
import json
from collections import Counter
from itertools import combinations
from math import comb, factorial
from pathlib import Path
from typing import Literal

import numpy as np
from scipy.stats import spearmanr, wasserstein_distance

from dp.utils.risk import _scores_to_inverse_probs


Dataset = Literal["db_bio", "tab", "rat_bench"]

DATASETS: list[Dataset] = ["db_bio", "tab", "rat_bench"]

DATA_DIR = Path("data")


ShapRecord = dict[str, list]  # {"offsets": list[tuple[int, int]], "scores": list[float]}


def load_shap(path: Path) -> dict[str, ShapRecord]:
    """Load a shap.jsonl file, canonicalizing each record's tokens into text
    order (sorted by offset start).

    risk.py's `sort_by: scores` writes offsets/scores sorted by score
    descending (by design — that's the traversal order anonymization masks
    tokens in). Two different explainer runs over the same text therefore
    save the SAME set of tokens in DIFFERENT list orders. Any code comparing
    `scores[i]` to `scores[i]` across two files without first re-deriving true
    token identity from `offsets` would be comparing "i-th highest score in
    run A" to "i-th highest score in run B" — not the same token. Sorting by
    offset here fixes that for every downstream consumer.
    """
    records: dict[str, ShapRecord] = {}
    with path.open() as f:
        for line in f:
            entry = json.loads(line)
            offsets = [tuple(o) for o in entry["offsets"]]
            scores = entry["scores"]
            order = sorted(range(len(offsets)), key=lambda i: offsets[i][0])
            records[entry["uid"]] = {
                "offsets": [offsets[i] for i in order],
                "scores": [scores[i] for i in order],
            }
    return records


def flatten_scores(shap: dict[str, ShapRecord], uids: list[str]) -> np.ndarray:
    return np.array([score for uid in uids for score in shap[uid]["scores"]])


def per_record_alignment_metrics(
    a: dict[str, ShapRecord],
    b: dict[str, ShapRecord],
    uids: list[str],
) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray, int]:
    """Position-aligned per-record distances: MAE (Mean Absolute Error), RMSE
    (Root Mean Squared Error), and MedAE (Median Absolute Error), computed
    only over records whose offset sequences match exactly between `a` and
    `b` (same tokenization of the same text). Records with mismatched offsets
    (different length/tokenization) are skipped and counted rather than
    silently misaligned.

    MAE is the primary metric — it weighs every token's disagreement
    proportionally. RMSE would let a single large per-token swing dominate the
    whole score; MedAE is reported separately as a companion that stays
    grounded in the bulk of the per-token differences even when one outlier
    token would otherwise dominate.
    """
    used_uids: list[str] = []
    mae_values: list[float] = []
    rmse_values: list[float] = []
    medae_values: list[float] = []
    skipped = 0
    for uid in uids:
        a_rec, b_rec = a[uid], b[uid]
        if not a_rec["offsets"] or a_rec["offsets"] != b_rec["offsets"]:
            skipped += 1
            continue
        diff = np.abs(np.asarray(a_rec["scores"], dtype=float) - np.asarray(b_rec["scores"], dtype=float))
        used_uids.append(uid)
        mae_values.append(float(diff.mean()))
        rmse_values.append(float(np.sqrt((diff ** 2).mean())))
        medae_values.append(float(np.median(diff)))
    return (
        used_uids,
        np.asarray(mae_values, dtype=float),
        np.asarray(rmse_values, dtype=float),
        np.asarray(medae_values, dtype=float),
        skipped,
    )


def per_record_spearman(
    full: dict[str, ShapRecord],
    exact: dict[str, ShapRecord],
    uids: list[str],
) -> tuple[np.ndarray, int]:
    correlations: list[float] = []
    skipped = 0
    for uid in uids:
        full_rec, exact_rec = full[uid], exact[uid]
        if not full_rec["offsets"] or full_rec["offsets"] != exact_rec["offsets"]:
            skipped += 1
            continue
        full_scores = np.asarray(full_rec["scores"], dtype=float)
        exact_scores = np.asarray(exact_rec["scores"], dtype=float)
        if len(full_scores) < 2:
            skipped += 1
            continue
        corr = spearmanr(full_scores, exact_scores).correlation
        if np.isfinite(corr):
            correlations.append(float(corr))
    return np.asarray(correlations, dtype=float), skipped


def per_record_topk_jaccard(
    a: dict[str, ShapRecord],
    b: dict[str, ShapRecord],
    uids: list[str],
    k: int,
) -> tuple[np.ndarray, int]:
    """Per-record top-k Jaccard overlap: the k highest-scoring (highest-risk,
    per the traversal-order convention in `load_shap`) tokens in `a` versus
    in `b`, as |intersection| / |union| of the two top-k index sets. Only
    defined over records with matching offsets and at least k tokens;
    everything else is skipped and counted.
    """
    jaccards: list[float] = []
    skipped = 0
    for uid in uids:
        a_rec, b_rec = a[uid], b[uid]
        if not a_rec["offsets"] or a_rec["offsets"] != b_rec["offsets"] or len(a_rec["offsets"]) < k:
            skipped += 1
            continue
        a_top = set(np.argsort(np.asarray(a_rec["scores"], dtype=float))[::-1][:k].tolist())
        b_top = set(np.argsort(np.asarray(b_rec["scores"], dtype=float))[::-1][:k].tolist())
        jaccards.append(len(a_top & b_top) / len(a_top | b_top))
    return np.asarray(jaccards, dtype=float), skipped


def allocation_dispersion(
    shap: dict[str, ShapRecord],
    uids: list[str],
    *,
    temperature: float = 0.1,
) -> dict[str, float]:
    stds: list[float] = []
    maxes: list[float] = []
    entropies: list[float] = []
    for uid in uids:
        scores = np.asarray(shap[uid]["scores"], dtype=float)
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


def analyze(dataset: Dataset, full_vs_exact: bool = False, bart_vs_nobart: bool = False) -> None:
    if full_vs_exact:
        full_path = DATA_DIR / dataset / "tri_risk" / "shap.jsonl"
        exact_path = DATA_DIR / dataset / "tri_risk" / "shap_subset.jsonl"
    elif bart_vs_nobart:
        full_path = DATA_DIR / dataset / "tri_risk" / "shap.jsonl"
        exact_path = DATA_DIR / dataset / "tri_risk" / "shap_nobart.jsonl"

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

    used_uids, per_record_mae, per_record_rmse, per_record_medae, skipped_align = per_record_alignment_metrics(
        full, exact, common_uids
    )
    print(f"Per-record offset-aligned distance metrics ({skipped_align} records skipped due to offset mismatch):")
    print(f"  MAE   — mean: {per_record_mae.mean():.6f}, std: {per_record_mae.std():.6f}, "
          f"min: {per_record_mae.min():.6f}, max: {per_record_mae.max():.6f}")
    print(f"  RMSE  — mean: {per_record_rmse.mean():.6f}")
    print(f"  MedAE — mean: {per_record_medae.mean():.6f}")

    top_k = 10
    top_indices = np.argsort(per_record_mae)[::-1][:top_k]
    print(f"  Top {top_k} UIDs by MAE:")
    for i in top_indices:
        print(f"    {used_uids[i]}: {per_record_mae[i]:.6f}")

    rank_corr, skipped = per_record_spearman(full, exact, common_uids)
    if rank_corr.size:
        print("Per-record Spearman's rank correlation coefficient:")
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


# --- "groups" mode: N models split into named groups (e.g. bart vs nobart seeds),
# compared pairwise via the offset-aligned per-record MAE above, then tested for
# whether cross-group pairs are more distant than within-group pairs. ---


def load_group_shap(
    groups: dict[str, list[Path]],
) -> tuple[dict[str, dict[str, ShapRecord]], dict[str, str], list[str]]:
    shap_by_model: dict[str, dict[str, ShapRecord]] = {}
    group_of: dict[str, str] = {}
    model_ids: list[str] = []
    for group_name, paths in groups.items():
        for path in paths:
            model_id = f"{group_name}:{path.stem}"
            shap_by_model[model_id] = load_shap(path)
            group_of[model_id] = group_name
            model_ids.append(model_id)
    return shap_by_model, group_of, model_ids


def pairwise_distance_matrix(
    shap_by_model: dict[str, dict[str, ShapRecord]],
    model_ids: list[str],
    common_uids: list[str],
    jaccard_k: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    n = len(model_ids)
    mae_matrix = np.zeros((n, n), dtype=float)
    rmse_matrix = np.zeros((n, n), dtype=float)
    medae_matrix = np.zeros((n, n), dtype=float)
    spearman_matrix = np.eye(n, dtype=float)
    jaccard_matrix = np.eye(n, dtype=float)
    total_skipped = 0
    for i in range(n):
        for j in range(i + 1, n):
            a, b = shap_by_model[model_ids[i]], shap_by_model[model_ids[j]]
            used_uids, mae, rmse, medae, skipped = per_record_alignment_metrics(a, b, common_uids)
            if mae.size == 0:
                raise ValueError(
                    f"No offset-aligned records between {model_ids[i]} and {model_ids[j]} "
                    "(tokenization/offsets differ — check they explain the same result_in text)"
                )
            mae_matrix[i, j] = mae_matrix[j, i] = float(mae.mean())
            rmse_matrix[i, j] = rmse_matrix[j, i] = float(rmse.mean())
            medae_matrix[i, j] = medae_matrix[j, i] = float(medae.mean())
            total_skipped += skipped

            rank_corr, _ = per_record_spearman(a, b, common_uids)
            if rank_corr.size == 0:
                raise ValueError(f"No records with >=2 tokens to compute Spearman for {model_ids[i]} vs {model_ids[j]}")
            spearman_matrix[i, j] = spearman_matrix[j, i] = float(rank_corr.mean())

            jaccard, _ = per_record_topk_jaccard(a, b, common_uids, jaccard_k)
            if jaccard.size == 0:
                raise ValueError(f"No records with >={jaccard_k} tokens to compute top-{jaccard_k} Jaccard for {model_ids[i]} vs {model_ids[j]}")
            jaccard_matrix[i, j] = jaccard_matrix[j, i] = float(jaccard.mean())
    return mae_matrix, rmse_matrix, medae_matrix, spearman_matrix, jaccard_matrix, total_skipped


def within_cross_split(
    model_ids: list[str],
    group_of: dict[str, str],
    matrix: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    n = len(model_ids)
    within: list[float] = []
    cross: list[float] = []
    for i in range(n):
        for j in range(i + 1, n):
            target = within if group_of[model_ids[i]] == group_of[model_ids[j]] else cross
            target.append(matrix[i, j])
    return np.asarray(within, dtype=float), np.asarray(cross, dtype=float)


def within_group_by_group(
    model_ids: list[str],
    group_of: dict[str, str],
    matrix: np.ndarray,
) -> dict[str, np.ndarray]:
    """Within-group pairwise distances, kept separate per group (unlike
    within_cross_split, which pools every group's own pairs into one "within"
    bucket). Lets you check whether one group's models are internally more
    scattered than the other's, rather than only a single pooled figure.
    """
    n = len(model_ids)
    by_group: dict[str, list[float]] = {g: [] for g in set(group_of.values())}
    for i in range(n):
        for j in range(i + 1, n):
            gi, gj = group_of[model_ids[i]], group_of[model_ids[j]]
            if gi == gj:
                by_group[gi].append(matrix[i, j])
    return {g: np.asarray(v, dtype=float) for g, v in by_group.items()}


MAX_EXACT_ASSIGNMENTS = 2_000_000


def _assignment_count(n: int, group_sizes: list[int]) -> int:
    total = 1
    remaining = n
    for size in group_sizes:
        total *= comb(remaining, size)
        remaining -= size
    return total


def _multiset_assignments(n: int, group_sizes: list[int]):
    """Yield every distinct way to split n indices into groups of the given
    sizes, as a length-n list of group indices per position.
    """
    def rec(remaining: list[int], sizes: list[int]):
        if len(sizes) == 1:
            yield [remaining]
            return
        for combo in combinations(remaining, sizes[0]):
            rest = [i for i in remaining if i not in combo]
            for tail in rec(rest, sizes[1:]):
                yield [list(combo)] + tail

    for groups in rec(list(range(n)), group_sizes):
        labels = [0] * n
        for group_index, idxs in enumerate(groups):
            for i in idxs:
                labels[i] = group_index
        yield labels


def _label_symmetry_factor(group_sizes: list[int]) -> int:
    """Number of label assignments that are guaranteed to tie any given one,
    purely from renaming which group index is called '0', '1', etc. — e.g.
    with two equal-sized groups, swapping which side is "0" never changes
    which pairs are within- vs. cross-group, so every split is counted twice
    among the enumerated assignments. This sets the true resolution floor
    (min count of "at least as extreme" assignments = this factor, however
    extreme the data), which is 1 only when all group sizes are distinct.
    """
    factor = 1
    for count in Counter(group_sizes).values():
        factor *= factorial(count)
    return factor


def group_label_permutation_test(
    model_ids: list[str],
    group_of: dict[str, str],
    matrix: np.ndarray,
) -> tuple[float, float, int, float]:
    """Exact permutation test on group labels: is the observed
    mean(cross-group distance) - mean(within-group distance) more extreme
    than under every possible relabeling of the same (fixed) distance matrix
    that preserves the original group sizes? With only a handful of models,
    the full set of label assignments (n! / (k1! k2! ...)) is cheap to
    enumerate exactly, which avoids both the independence assumption of a
    t-test and the Monte Carlo noise/arbitrary sample count of a randomized
    permutation test — and makes the achievable minimum p-value explicit
    rather than hidden inside a chosen permutation count.
    """
    n = len(model_ids)
    idx_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    group_names = sorted(set(group_of.values()))
    group_sizes = [sum(1 for m in model_ids if group_of[m] == g) for g in group_names]

    total_assignments = _assignment_count(n, group_sizes)
    if total_assignments > MAX_EXACT_ASSIGNMENTS:
        raise SystemExit(
            f"Exact permutation test would enumerate {total_assignments} label "
            f"assignments (> {MAX_EXACT_ASSIGNMENTS}); group sizes are too large "
            "for exact enumeration."
        )

    def stat(labels: list[int]) -> float:
        within: list[float] = []
        cross: list[float] = []
        for i, j in idx_pairs:
            (within if labels[i] == labels[j] else cross).append(matrix[i, j])
        return float(np.mean(cross) - np.mean(within))

    name_to_index = {name: i for i, name in enumerate(group_names)}
    observed_labels = [name_to_index[group_of[m]] for m in model_ids]
    observed = stat(observed_labels)

    extreme = sum(
        1 for labels in _multiset_assignments(n, group_sizes)
        if abs(stat(labels)) >= abs(observed)
    )
    p_value = extreme / total_assignments
    min_p_value = _label_symmetry_factor(group_sizes) / total_assignments
    return observed, p_value, total_assignments, min_p_value


def render_heatmap(
    model_ids: list[str],
    group_of: dict[str, str],
    matrix: np.ndarray,
    output_path: Path,
    decimals: int = 5,
) -> None:
    import matplotlib.pyplot as plt

    order = sorted(range(len(model_ids)), key=lambda i: (group_of[model_ids[i]], model_ids[i]))
    ordered_ids = [model_ids[i] for i in order]
    ordered_matrix = matrix[np.ix_(order, order)]

    fig, ax = plt.subplots(figsize=(0.6 * len(model_ids) + 2, 0.6 * len(model_ids) + 2))
    im = ax.imshow(ordered_matrix, cmap="viridis")
    ax.set_xticks(range(len(ordered_ids)))
    ax.set_yticks(range(len(ordered_ids)))
    ax.set_xticklabels(ordered_ids, rotation=90, fontsize=8)
    ax.set_yticklabels(ordered_ids, fontsize=8)
    for i in range(len(ordered_ids)):
        for j in range(len(ordered_ids)):
            ax.text(j, i, f"{ordered_matrix[i, j]:.{decimals}f}", ha="center", va="center",
                     color="white", fontsize=6)

    group_sizes = [len(list(g)) for _, g in _groupby_preserve_order(
        [group_of[m] for m in ordered_ids]
    )]
    boundary = -0.5
    for size in group_sizes[:-1]:
        boundary += size
        ax.axhline(boundary, color="red", linewidth=1.5)
        ax.axvline(boundary, color="red", linewidth=1.5)

    ax.set_title("Pairwise Mean Absolute Error (offset-aligned SHAP scores)")
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def render_tsne(
    model_ids: list[str],
    group_of: dict[str, str],
    matrix: np.ndarray,
    output_path: Path,
    seed: int,
    perplexity: float | None = None,
) -> None:
    """2D t-SNE projection of the models from their pairwise MAE distance
    matrix (metric="precomputed"), colored by group. With only a handful of
    models per group, this is illustrative rather than a rigorous clustering
    result — perplexity is capped well below its usual default (30) because
    sklearn requires it to be smaller than the number of points.
    """
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE

    n = len(model_ids)
    if perplexity is None:
        perplexity = max(1.0, min(30.0, (n - 1) / 3))

    coords = TSNE(
        n_components=2,
        metric="precomputed",
        init="random",
        perplexity=perplexity,
        random_state=seed,
    ).fit_transform(matrix)

    fig, ax = plt.subplots(figsize=(6, 6))
    group_names = sorted(set(group_of.values()))
    cmap = plt.get_cmap("tab10")
    for gi, group_name in enumerate(group_names):
        idx = [i for i, m in enumerate(model_ids) if group_of[m] == group_name]
        ax.scatter(coords[idx, 0], coords[idx, 1], label=group_name, color=cmap(gi), s=80)
    for i, model_id in enumerate(model_ids):
        ax.annotate(model_id, (coords[i, 0], coords[i, 1]), fontsize=7,
                    xytext=(4, 4), textcoords="offset points")

    ax.set_title(f"t-SNE of pairwise SHAP MAE distances (perplexity={perplexity:.1f})")
    ax.legend()
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _groupby_preserve_order(labels: list[str]) -> list[tuple[str, list[str]]]:
    result: list[tuple[str, list[str]]] = []
    for label in labels:
        if result and result[-1][0] == label:
            result[-1][1].append(label)
        else:
            result.append((label, [label]))
    return result


def analyze_groups(
    groups: dict[str, list[Path]],
    *,
    alpha: float,
    seed: int,
    jaccard_k: int,
    heatmap_out: Path | None,
    heatmap_decimals: int,
    tsne_out: Path | None,
    tsne_perplexity: float | None,
    json_out: Path | None,
) -> None:
    shap_by_model, group_of, model_ids = load_group_shap(groups)

    common_uids = sorted(set.intersection(*(set(shap.keys()) for shap in shap_by_model.values())))
    print(f"Models: {len(model_ids)} across {len(groups)} groups {list(groups.keys())}")
    print(f"Common records across all models: {len(common_uids)}")
    if not common_uids:
        raise SystemExit("No records common to every model — check that all shap.jsonl files cover the same dataset")

    mae_matrix, rmse_matrix, medae_matrix, spearman_matrix, jaccard_matrix, total_skipped = pairwise_distance_matrix(
        shap_by_model, model_ids, common_uids, jaccard_k
    )
    print(f"Records skipped due to offset mismatch (summed across all model pairs): {total_skipped}")

    metric_matrices = {
        "MAE": mae_matrix,
        "RMSE": rmse_matrix,
        "MedAE": medae_matrix,
        "Spearman": spearman_matrix,
        f"Jaccard@{jaccard_k}": jaccard_matrix,
    }
    within_cross_by_metric: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    within_by_group_by_metric: dict[str, dict[str, np.ndarray]] = {}

    for metric_name, matrix in metric_matrices.items():
        w, c = within_cross_split(model_ids, group_of, matrix)
        within_cross_by_metric[metric_name] = (w, c)
        print(f"\nWithin-group pairs ({metric_name}): n={len(w)}, mean={w.mean():.6f}, std={w.std():.6f}")
        print(f"Cross-group pairs ({metric_name}):  n={len(c)}, mean={c.mean():.6f}, std={c.std():.6f}")

        by_group = within_group_by_group(model_ids, group_of, matrix)
        within_by_group_by_metric[metric_name] = by_group
        print(f"Within-group pairs by group ({metric_name}):")
        for group_name in sorted(by_group):
            vals = by_group[group_name]
            print(f"  {group_name}: n={len(vals)}, mean={vals.mean():.6f}, std={vals.std():.6f}")

    # Spearman and Jaccard are similarity metrics (higher = more alike), unlike
    # MAE/RMSE/MedAE (higher = more different); the permutation test uses
    # |mean(cross) - mean(within)| so this sign flip doesn't need special-casing.
    permutation_metrics = ["MAE", "Spearman", f"Jaccard@{jaccard_k}"]
    permutation_results: dict[str, tuple[float, float, int, float]] = {}
    for metric_name in permutation_metrics:
        observed, perm_p, total_assignments, min_p = group_label_permutation_test(
            model_ids, group_of, metric_matrices[metric_name]
        )
        permutation_results[metric_name] = (observed, perm_p, total_assignments, min_p)
        print(f"\nGroup-label permutation test on {metric_name} (exact, {total_assignments} label assignments enumerated):")
        print(f"  observed mean(cross) - mean(within) = {observed:.6f}")
        print(f"  p-value = {perm_p:.6f} ({'significant' if perm_p < alpha else 'not significant'} at alpha={alpha}; "
              f"minimum achievable p-value at this group size = {min_p:.6f})")

    if heatmap_out is not None:
        render_heatmap(model_ids, group_of, mae_matrix, heatmap_out, decimals=heatmap_decimals)
        print(f"\nHeatmap saved to {heatmap_out}")

    if tsne_out is not None:
        render_tsne(model_ids, group_of, mae_matrix, tsne_out, seed=seed, perplexity=tsne_perplexity)
        print(f"t-SNE plot saved to {tsne_out}")

    if json_out is not None:
        json_out.parent.mkdir(parents=True, exist_ok=True)
        json_out.write_text(json.dumps({
            "model_ids": model_ids,
            "group_of": group_of,
            "common_record_count": len(common_uids),
            "records_skipped_offset_mismatch": total_skipped,
            "mae_matrix": mae_matrix.tolist(),
            "rmse_matrix": rmse_matrix.tolist(),
            "medae_matrix": medae_matrix.tolist(),
            "spearman_matrix": spearman_matrix.tolist(),
            "jaccard_matrix": jaccard_matrix.tolist(),
            "jaccard_k": jaccard_k,
            "within_group_distances": {m: w.tolist() for m, (w, c) in within_cross_by_metric.items()},
            "cross_group_distances": {m: c.tolist() for m, (w, c) in within_cross_by_metric.items()},
            "within_group_distances_by_group": {
                m: {g: v.tolist() for g, v in by_group.items()}
                for m, by_group in within_by_group_by_metric.items()
            },
            "permutation_tests": {
                metric_name: {"observed_diff": observed, "p_value": perm_p,
                               "total_assignments": total_assignments,
                               "min_achievable_p_value": min_p}
                for metric_name, (observed, perm_p, total_assignments, min_p) in permutation_results.items()
            },
            "alpha": alpha,
        }, indent=2))
        print(f"Full results saved to {json_out}")


def _parse_group_arg(spec: list[str]) -> tuple[str, list[Path]]:
    if len(spec) < 2:
        raise argparse.ArgumentTypeError(
            "--group requires a name followed by at least one shap.jsonl path"
        )
    name, *paths = spec
    return name, [Path(p) for p in paths]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="mode")

    legacy_parser = subparsers.add_parser(
        "legacy", help="original full-vs-exact-subset comparison for db_bio/tab"
    )
    legacy_parser.add_argument(
        "--dataset", choices=DATASETS, default=None,
        help="restrict to a single dataset instead of running both db_bio and tab",
    )

    groups_parser = subparsers.add_parser(
        "groups", help="compare N models split into named groups (e.g. bart vs nobart seeds)"
    )
    groups_parser.add_argument(
        "--group", dest="groups", action="append", nargs="+", required=True,
        metavar=("NAME", "PATH"),
        help="--group <name> <path1> [path2 ...], repeatable for each group",
    )
    groups_parser.add_argument("--alpha", type=float, default=0.05)
    groups_parser.add_argument("--seed", type=int, default=0)
    groups_parser.add_argument("--jaccard-k", type=int, default=5,
                                help="top-k tokens by SHAP score used for the Jaccard overlap metric")
    groups_parser.add_argument("--heatmap-out", type=Path, default=None)
    groups_parser.add_argument("--heatmap-decimals", type=int, default=5)
    groups_parser.add_argument("--tsne-out", type=Path, default=None)
    groups_parser.add_argument("--tsne-perplexity", type=float, default=None)
    groups_parser.add_argument("--json-out", type=Path, default=None)

    args = parser.parse_args()

    if args.mode is None or args.mode == "legacy":
        datasets = [args.dataset] if getattr(args, "dataset", None) else DATASETS
        for dataset in datasets:
            analyze(dataset)
        return

    groups: dict[str, list[Path]] = {}
    for spec in args.groups:
        name, paths = _parse_group_arg(spec)
        if name in groups:
            raise SystemExit(f"Duplicate --group name: {name}")
        groups[name] = paths
    if len(groups) < 2:
        raise SystemExit("groups mode requires at least 2 --group arguments")

    analyze_groups(
        groups,
        alpha=args.alpha,
        seed=args.seed,
        jaccard_k=args.jaccard_k,
        heatmap_out=args.heatmap_out,
        heatmap_decimals=args.heatmap_decimals,
        tsne_out=args.tsne_out,
        tsne_perplexity=args.tsne_perplexity,
        json_out=args.json_out,
    )


if __name__ == "__main__":
    main()
