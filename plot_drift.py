from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from scipy.interpolate import interp1d

LOGS_DIR = Path("logs/risk_drift")
IMAGES_DIR = Path("images/drift")
W_SCALE = 1e4
N_COMMON = 100
T_ORDER: dict[str, int] = {"1": 0, "2": 1, "5": 2, "10": 3, "inf": 4}
T_DISP: dict[str, str] = {"1": "1", "2": "2", "5": "5", "10": "10", "inf": r"\infty"}

# Pairs ordered from semantically closest to furthest t-distance.
# Finite pairs ordered by log₂(t_b/t_a); inf pairs ordered by descending t_a
# (10_vs_inf is the least extreme, 1_vs_inf is the most).
PAIR_ORDER = [
    "1_vs_2", "5_vs_10",            # log₂ ratio = 1.0
    "2_vs_5",                        # log₂ ratio ≈ 1.32
    "1_vs_5", "2_vs_10",            # log₂ ratio ≈ 2.32
    "1_vs_10",                       # log₂ ratio ≈ 3.32
    "10_vs_inf", "5_vs_inf", "2_vs_inf", "1_vs_inf",
]


def pair_label(pair: str, t_disp: dict[str, str]) -> str:
    a, b = pair.split("_vs_")
    return rf"$t={t_disp[a]}$ vs $t={t_disp[b]}$"


def pair_colors(n: int) -> list:
    return [cm.plasma(v) for v in np.linspace(0.1, 0.85, n)]


def load_drift(
    path: Path,
) -> tuple[list[str], np.ndarray, list[str], dict[str, np.ndarray], dict[str, np.ndarray]]:
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    uids = [r["uid"] for r in records]
    n_tokens = np.array([r["n_tokens"] for r in records])
    t_values: list[str] = records[0]["t_values"]
    pair_keys = list(records[0]["steps"][0]["pairs"].keys())
    wasserstein: dict[str, np.ndarray] = {
        p: np.array([[s["pairs"][p]["wasserstein"] for s in r["steps"]] for r in records])
        for p in pair_keys
    }
    jaccard: dict[str, np.ndarray] = {
        p: np.array([[s["pairs"][p]["jaccard"] for s in r["steps"]] for r in records])
        for p in pair_keys
    }
    return uids, n_tokens, t_values, wasserstein, jaccard


def interpolate_rows(matrix: np.ndarray, x_per_row: np.ndarray, x_common: np.ndarray) -> np.ndarray:
    out = np.zeros((len(matrix), len(x_common)))
    for i, (row, x) in enumerate(zip(matrix, x_per_row)):
        f = interp1d(x, row, bounds_error=False, fill_value=(row[0], row[-1]))
        out[i] = f(x_common)
    return out


def plot_trajectories(
    pair_keys: list[str],
    w_interp: dict[str, np.ndarray],
    j_interp: dict[str, np.ndarray],
    frac_common: np.ndarray,
    max_steps: int,
    t_disp: dict[str, str],
    out_path: Path,
) -> None:
    ordered = [p for p in PAIR_ORDER if p in pair_keys]
    ordered += [p for p in pair_keys if p not in ordered]
    colors = pair_colors(len(ordered))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    for pair, color in zip(ordered, colors):
        label = pair_label(pair, t_disp)
        w_med = np.median(w_interp[pair], axis=0)
        j_mean = j_interp[pair].mean(axis=0)
        axes[0].plot(frac_common, w_med * W_SCALE, color=color, linewidth=1.6, label=label)
        axes[1].plot(frac_common, j_mean, color=color, linewidth=1.6, label=label)

    axes[0].set_xlabel("Fraction of context removed")
    axes[0].set_ylabel(r"Wasserstein distance ($\times 10^{-4}$)")
    axes[0].set_title(rf"SHAP Distribution Divergence (max steps $= {max_steps}$)")
    axes[0].set_ylim(bottom=0)
    axes[0].yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:.1f}"))
    axes[0].legend(frameon=False, fontsize=8, ncol=2)
    axes[0].spines["top"].set_visible(False)
    axes[0].spines["right"].set_visible(False)

    axes[1].set_xlabel("Fraction of context removed")
    axes[1].set_ylabel("Jaccard similarity")
    axes[1].set_title(rf"Top-$k$ Token Overlap (max steps $= {max_steps}$)")
    axes[1].set_ylim(-0.02, 1.05)
    axes[1].legend(frameon=False, fontsize=8, ncol=2)
    axes[1].spines["top"].set_visible(False)
    axes[1].spines["right"].set_visible(False)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"saved {out_path}")


def plot_heatmaps(
    pair_keys: list[str],
    t_values: list[str],
    wasserstein: dict[str, np.ndarray],
    jaccard: dict[str, np.ndarray],
    max_steps: int,
    t_disp: dict[str, str],
    out_path: Path,
) -> None:
    ordered_t = sorted(t_values, key=lambda x: T_ORDER[x])
    n_t = len(ordered_t)
    t_labels = [rf"$t={t_disp[t]}$" for t in ordered_t]

    w_mat = np.full((n_t, n_t), np.nan)
    j_mat = np.full((n_t, n_t), np.nan)

    for pair in pair_keys:
        a, b = pair.split("_vs_")
        i, j = T_ORDER[a], T_ORDER[b]
        w_val = float(np.mean(wasserstein[pair][:, -1]))
        j_val = float(np.mean(jaccard[pair][:, -1]))
        w_mat[i, j] = w_val
        w_mat[j, i] = w_val
        j_mat[i, j] = j_val
        j_mat[j, i] = j_val

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    specs = [
        (axes[0], w_mat * W_SCALE, rf"Wasserstein ($\times 10^{{-4}}$, step $= {max_steps}$)", "Blues", ".2f"),
        (axes[1], j_mat, rf"Jaccard similarity (step $= {max_steps}$)", "RdYlGn", ".2f"),
    ]
    for ax, mat, title, cmap, fmt in specs:
        masked = np.ma.masked_invalid(mat)
        im = ax.imshow(masked, cmap=cmap, aspect="auto",
                       vmin=float(np.nanmin(mat)), vmax=float(np.nanmax(mat)))
        ax.set_xticks(range(n_t))
        ax.set_yticks(range(n_t))
        ax.set_xticklabels(t_labels, fontsize=9)
        ax.set_yticklabels(t_labels, fontsize=9)
        ax.set_title(title)
        fig.colorbar(im, ax=ax, shrink=0.8)
        thresh = float(np.nanmin(mat)) + 0.55 * (float(np.nanmax(mat)) - float(np.nanmin(mat)))
        for ii in range(n_t):
            for jj in range(n_t):
                if not np.isnan(mat[ii, jj]):
                    text_color = "white" if mat[ii, jj] > thresh else "black"
                    ax.text(jj, ii, format(mat[ii, jj], fmt),
                            ha="center", va="center", fontsize=8, color=text_color)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"saved {out_path}")


def plot(drift_path: Path, out_dir: Path) -> None:
    uids, n_tokens, t_values, wasserstein, jaccard = load_drift(drift_path)
    pair_keys = list(wasserstein.keys())

    n_steps = next(iter(wasserstein.values())).shape[1]
    t_disp = {**T_DISP, "inf": str(n_steps)}
    steps = np.arange(n_steps)
    frac = steps[np.newaxis, :] / n_tokens[:, np.newaxis]
    frac_common = np.linspace(0, float(frac.max()), N_COMMON)

    w_interp = {p: interpolate_rows(wasserstein[p], frac, frac_common) for p in pair_keys}
    j_interp = {p: interpolate_rows(jaccard[p], frac, frac_common) for p in pair_keys}

    plot_trajectories(pair_keys, w_interp, j_interp, frac_common, n_steps, t_disp,
                      out_dir / "drift_by_t_distance.png")
    plot_heatmaps(pair_keys, t_values, wasserstein, jaccard, n_steps, t_disp,
                  out_dir / "drift_heatmap.png")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--variant", default=None)
    args = parser.parse_args()

    if args.dataset and args.variant:
        candidates = [LOGS_DIR / args.dataset / args.variant]
    elif args.dataset:
        candidates = sorted((LOGS_DIR / args.dataset).iterdir())
    else:
        candidates = sorted(p for d in LOGS_DIR.iterdir() for p in d.iterdir())

    for run_dir in candidates:
        drift_path = run_dir / "drift.jsonl"
        if not drift_path.exists():
            print(f"skipping {run_dir} (no drift.jsonl)")
            continue
        dataset, variant = run_dir.parent.name, run_dir.name
        plot(drift_path, IMAGES_DIR / dataset / variant)


if __name__ == "__main__":
    main()
