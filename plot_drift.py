from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from scipy.interpolate import interp1d

LOGS_DIR = Path("logs/risk_drift")
IMAGES_DIR = Path("images/drift")
CONFIGS_DIR = Path("configs/a3_risk_drift")

C_W_MAIN = "#2563EB"
C_W_BAND = "#BFDBFE"
C_J_MAIN = "#16A34A"
C_J_BAND = "#BBF7D0"
C_BOUND = "#DC2626"
W_SCALE = 1e4


def load_drift(path: Path) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray]:
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    uids = [r["uid"] for r in records]
    n_tokens = np.array([r["n_tokens"] for r in records])
    wasserstein = np.array([[s["wasserstein"] for s in r["steps"]] for r in records])
    jaccard = np.array([[s["jaccard"] for s in r["steps"]] for r in records])
    return uids, n_tokens, wasserstein, jaccard


def load_shap_mads(shap_path: Path, uids: list[str]) -> np.ndarray | None:
    uid_set = set(uids)
    mads: dict[str, float] = {}
    with shap_path.open(encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r["uid"] in uid_set:
                scores = np.array(r["scores"])
                mads[r["uid"]] = float(np.mean(np.abs(scores - scores.mean())))
    matched = [mads[uid] for uid in uids if uid in mads]
    return np.array(matched) if matched else None


def shap_path_for(dataset: str, variant: str) -> Path | None:
    config_path = CONFIGS_DIR / f"{dataset}_{variant}.yaml"
    if not config_path.exists():
        return None
    import yaml
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    return Path(cfg["shap_in"]) if "shap_in" in cfg else None


def interpolate_to_common(
    matrix: np.ndarray, x_per_row: np.ndarray, x_common: np.ndarray
) -> np.ndarray:
    out = np.zeros((len(matrix), len(x_common)))
    for i, (row, x) in enumerate(zip(matrix, x_per_row)):
        f = interp1d(x, row, bounds_error=False, fill_value=(row[0], row[-1]))
        out[i] = f(x_common)
    return out


def plot(drift_path: Path, out_path: Path, shap_path: Path | None = None) -> None:
    uids, n_tokens, wasserstein, jaccard = load_drift(drift_path)

    n_steps = wasserstein.shape[1]
    steps = np.arange(n_steps)
    frac = steps[np.newaxis, :] / n_tokens[:, np.newaxis]
    frac_common = np.linspace(0, frac.max(), 100)

    w_interp = interpolate_to_common(wasserstein, frac, frac_common)
    j_interp = interpolate_to_common(jaccard, frac, frac_common)

    w_p25 = np.percentile(w_interp, 45, axis=0)
    w_p75 = np.percentile(w_interp, 55, axis=0)

    w_median = np.median(w_interp, axis=0)

    j_mean = j_interp.mean(axis=0)
    j_p25 = np.percentile(j_interp, 25, axis=0)
    j_p75 = np.percentile(j_interp, 75, axis=0)

    shap_mads: np.ndarray | None = None
    if shap_path is not None and shap_path.exists():
        shap_mads = load_shap_mads(shap_path, uids)

    w_bound = float(shap_mads.mean()) if shap_mads is not None else None
    w_top = (w_bound * W_SCALE * 1.1) if w_bound is not None else (w_p75.max() * W_SCALE * 1.3)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    ax = axes[0]
    ax.fill_between(frac_common, w_p25 * W_SCALE, w_p75 * W_SCALE, color=C_W_BAND, alpha=0.7, label="45–55th pct")
    ax.plot(frac_common, w_median * W_SCALE, color=C_W_MAIN, linewidth=2.0, linestyle="--", label="Median")
    if w_bound is not None:
        ax.axhline(w_bound * W_SCALE, color=C_BOUND, linewidth=1.5, linestyle="--",
                   label=r"$\overline{\mathrm{MAD}}(\phi)$")
    ax.set_xlabel("Fraction of context removed")
    ax.set_ylabel(r"Wasserstein distance ($\times 10^{-4}$)")
    ax.set_title("SHAP Score Distribution Drift")
    ax.set_ylim(bottom=0, top=w_top)
    ax.legend(frameon=False, fontsize=9)
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:.1f}"))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    j_skip = 3
    ax = axes[1]
    ax.fill_between(frac_common[j_skip:], j_p25[j_skip:], j_p75[j_skip:], color=C_J_BAND, alpha=0.7, label="IQR")
    ax.plot(frac_common[j_skip:], j_mean[j_skip:], color=C_J_MAIN, linewidth=2.0, label="Mean")
    ax.set_xlabel("Fraction of context removed")
    ax.set_ylabel("Jaccard similarity")
    ax.set_title("Top-$k$ Token Selection Overlap")
    ax.set_ylim(-0.02, 1.05)
    ax.legend(frameon=False, fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"saved {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--variant", default=None)
    parser.add_argument("--shap", default=None, help="path to shap.jsonl for W1 upper bound")
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
        shap_path = Path(args.shap) if args.shap else shap_path_for(dataset, variant)
        out_path = IMAGES_DIR / dataset / variant / "drift_wasserstein_jaccard.pdf"
        plot(drift_path, out_path, shap_path=shap_path)


if __name__ == "__main__":
    main()
