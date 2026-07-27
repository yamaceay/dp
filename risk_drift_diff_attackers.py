import argparse
import json
from pathlib import Path
from typing import Literal

import numpy as np
from scipy.stats import spearmanr, ttest_ind, wasserstein_distance

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


# --- "groups" mode: N models split into named groups (e.g. bart vs nobart seeds),
# compared pairwise via the same per-record Wasserstein distance used above, then
# tested for whether cross-group pairs are more distant than within-group pairs. ---


def load_group_shap(
    groups: dict[str, list[Path]],
) -> tuple[dict[str, dict[str, list[float]]], dict[str, str], list[str]]:
    shap_by_model: dict[str, dict[str, list[float]]] = {}
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
    shap_by_model: dict[str, dict[str, list[float]]],
    model_ids: list[str],
    common_uids: list[str],
) -> np.ndarray:
    n = len(model_ids)
    matrix = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i + 1, n):
            distances = per_record_distances(
                shap_by_model[model_ids[i]], shap_by_model[model_ids[j]], common_uids
            )
            matrix[i, j] = matrix[j, i] = float(distances.mean())
    return matrix


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


def group_label_permutation_test(
    model_ids: list[str],
    group_of: dict[str, str],
    matrix: np.ndarray,
    n_permutations: int,
    seed: int,
) -> tuple[float, float]:
    """Two-sided permutation test on group labels: is the observed
    mean(cross-group distance) - mean(within-group distance) more extreme than
    under random relabelings of the same (fixed) distance matrix? This sidesteps
    the fact that pairwise distances aren't independent samples, which a plain
    t-test on within/cross values would silently assume away.
    """
    n = len(model_ids)
    idx_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    labels = np.array([group_of[m] for m in model_ids])

    def stat(current_labels: np.ndarray) -> float:
        within: list[float] = []
        cross: list[float] = []
        for i, j in idx_pairs:
            (within if current_labels[i] == current_labels[j] else cross).append(matrix[i, j])
        return float(np.mean(cross) - np.mean(within))

    observed = stat(labels)
    rng = np.random.default_rng(seed)
    perm_stats = np.empty(n_permutations, dtype=float)
    for k in range(n_permutations):
        perm_stats[k] = stat(rng.permutation(labels))
    p_value = float(np.mean(np.abs(perm_stats) >= abs(observed)))
    return observed, p_value


def welch_ttest_and_power(
    within: np.ndarray,
    cross: np.ndarray,
    alpha: float,
) -> dict[str, float]:
    """Standard two-sample t-test + Cohen's d + achieved power at `alpha`, treating
    within/cross distances as independent samples. This is an approximation (the
    independence assumption doesn't strictly hold — see group_label_permutation_test
    for the assumption-free version) but is reported alongside it since it's the
    familiar/expected significance-test format.
    """
    from statsmodels.stats.power import TTestIndPower

    t_stat, p_value = ttest_ind(cross, within, equal_var=False)
    pooled_std = np.sqrt((cross.var(ddof=1) + within.var(ddof=1)) / 2)
    cohens_d = float((cross.mean() - within.mean()) / pooled_std) if pooled_std > 0 else 0.0
    achieved_power = float(
        TTestIndPower().power(
            effect_size=abs(cohens_d),
            nobs1=len(cross),
            ratio=len(within) / len(cross) if len(cross) else 1.0,
            alpha=alpha,
        )
    )
    return {
        "t_stat": float(t_stat),
        "p_value": float(p_value),
        "cohens_d": cohens_d,
        "achieved_power": achieved_power,
    }


def render_heatmap(
    model_ids: list[str],
    group_of: dict[str, str],
    matrix: np.ndarray,
    output_path: Path,
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
            ax.text(j, i, f"{ordered_matrix[i, j]:.3f}", ha="center", va="center",
                     color="white", fontsize=6)

    group_sizes = [len(list(g)) for _, g in _groupby_preserve_order(
        [group_of[m] for m in ordered_ids]
    )]
    boundary = -0.5
    for size in group_sizes[:-1]:
        boundary += size
        ax.axhline(boundary, color="red", linewidth=1.5)
        ax.axvline(boundary, color="red", linewidth=1.5)

    ax.set_title("Pairwise mean per-record Wasserstein distance (SHAP scores)")
    fig.colorbar(im, ax=ax, shrink=0.8)
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
    n_permutations: int,
    seed: int,
    heatmap_out: Path | None,
    json_out: Path | None,
) -> None:
    shap_by_model, group_of, model_ids = load_group_shap(groups)

    common_uids = sorted(set.intersection(*(set(shap.keys()) for shap in shap_by_model.values())))
    print(f"Models: {len(model_ids)} across {len(groups)} groups {list(groups.keys())}")
    print(f"Common records across all models: {len(common_uids)}")
    if not common_uids:
        raise SystemExit("No records common to every model — check that all shap.jsonl files cover the same dataset")

    matrix = pairwise_distance_matrix(shap_by_model, model_ids, common_uids)
    within, cross = within_cross_split(model_ids, group_of, matrix)

    print(f"\nWithin-group pairs: n={len(within)}, mean={within.mean():.6f}, std={within.std():.6f}")
    print(f"Cross-group pairs:  n={len(cross)}, mean={cross.mean():.6f}, std={cross.std():.6f}")

    observed, perm_p = group_label_permutation_test(model_ids, group_of, matrix, n_permutations, seed)
    print(f"\nGroup-label permutation test ({n_permutations} permutations, seed={seed}):")
    print(f"  observed mean(cross) - mean(within) = {observed:.6f}")
    print(f"  p-value = {perm_p:.6f} ({'significant' if perm_p < alpha else 'not significant'} at alpha={alpha})")

    ttest_result = welch_ttest_and_power(within, cross, alpha)
    print(f"\nWelch's t-test on within vs cross distances (independence assumption approximate):")
    print(f"  t = {ttest_result['t_stat']:.4f}, p = {ttest_result['p_value']:.6f} "
          f"({'significant' if ttest_result['p_value'] < alpha else 'not significant'} at alpha={alpha})")
    print(f"  Cohen's d = {ttest_result['cohens_d']:.4f}, achieved power = {ttest_result['achieved_power']:.4f}")

    if heatmap_out is not None:
        render_heatmap(model_ids, group_of, matrix, heatmap_out)
        print(f"\nHeatmap saved to {heatmap_out}")

    if json_out is not None:
        json_out.parent.mkdir(parents=True, exist_ok=True)
        json_out.write_text(json.dumps({
            "model_ids": model_ids,
            "group_of": group_of,
            "common_record_count": len(common_uids),
            "distance_matrix": matrix.tolist(),
            "within_group_distances": within.tolist(),
            "cross_group_distances": cross.tolist(),
            "permutation_test": {"observed_diff": observed, "p_value": perm_p,
                                  "n_permutations": n_permutations, "seed": seed},
            "welch_ttest": ttest_result,
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
    groups_parser.add_argument("--permutations", type=int, default=10000)
    groups_parser.add_argument("--seed", type=int, default=0)
    groups_parser.add_argument("--heatmap-out", type=Path, default=None)
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
        n_permutations=args.permutations,
        seed=args.seed,
        heatmap_out=args.heatmap_out,
        json_out=args.json_out,
    )


if __name__ == "__main__":
    main()
