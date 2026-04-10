from __future__ import annotations

import ast
import argparse
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

from plot_layout import create_panel_axes
from progress_utils import new_progress

MDS_DIR = Path("mds")
OUTPUT_DIR = Path("images/summary")

_PALETTE = [
    "#0072B2",
    "#D55E00",
    "#009E73",
    "#CC79A7",
    "#E69F00",
    "#56B4E9",
]
PRESIDIO_COLOR = "#8F2D56"
OTHER_THRESHOLD_PARAMS: frozenset[str] = frozenset({"k", "rho", "lambda"})

METRICS: list[str] = [
    "P_exact", "P_more", "P_full",
    "TRIR_exact", "TRIR_more", "TRIR_full",
    "U_acc", "U_mae",
    "SS_acc", "SS_mae",
    "D_bertscore", "D_cosine", "D_pp",
    "T_anon_avg_s",
]

FIXED_EPSILON_PANELS: list[float | None] = [None, 10, 25, 50, 100, 250]

X_AXIS_OPTIONS: list[str] = ["P_exact", "P_more", "P_full", "TRIR_exact", "TRIR_more", "TRIR_full"]
Y_AXIS_OPTIONS: list[str] = [
    "U_acc", "U_mae",
    "SS_acc", "SS_mae",
    "D_bertscore", "D_cosine", "D_pp",
    "T_anon_avg_s",
]

MAE_METRICS: frozenset[str] = frozenset({"U_ordinal_mae", "SS_ordinal_mae"})

MAE_RAW_COLUMN: dict[str, str] = {
    "U_ordinal_mae": "utility_utility_ordinal_raw_mae",
    "SS_ordinal_mae": "supervised_similarity_utility_ordinal_raw_mae",
}

METRIC_AXIS_LABELS: dict[str, str] = {
    "P_exact": "P_exact (MRR ↓ better)",
    "P_more": "P_more (MRR ↓ better)",
    "P_full": "P_full (MRR ↓ better)",
    "TRIR_exact": "TRIR_exact (Acc ↓ better)",
    "TRIR_more": "TRIR_more (Acc ↓ better)",
    "TRIR_full": "TRIR_full (Acc ↓ better)",
    "U_acc": "U_acc (Acc ↑ better)",
    "U_mae": "U_mae (MAE ↓ better)",
    "SS_acc": "SS_acc (Acc ↑ better)",
    "SS_mae": "SS_mae (MAE ↓ better)",
    "D_bertscore": "D_bertscore (↓ better)",
    "D_cosine": "D_cosine (↓ better)",
    "D_pp": "D_pp (↓ better)",
    "T_anon_avg_s": "Avg. anon time (s ↓ better)",
}

DATASET_LABELS: dict[str, str] = {"db_bio": "DB-Bio", "tab": "TAB"}


@dataclass(frozen=True)
class VariantSpec:
    method: str
    threshold_param: str
    label: str
    csv_suffix: str
    datasets: Optional[frozenset[str]]


MASKING_VARIANTS: list[VariantSpec] = [
    VariantSpec("risk_shap", "rho", "Risk+Pr (ρ)", "risk_maskers", None),
    VariantSpec("petre_shap", "k", "PETRE+Pr (k)", "simple_maskers_and_rewriters", None),
    VariantSpec("baroud", "lambda", "IPI (λ)", "simple_maskers_and_rewriters", frozenset({"tab"})),
]

DP_VARIANTS: list[VariantSpec] = [
    VariantSpec("dpmlm_shap", "k", "k-DP-MLM-X+Pr", "token_level_rewriters_with_threshold_k", None),
    VariantSpec("dpmlm_shap", "rho", "ρ-DP-MLM-X+Pr", "token_level_rewriters_with_threshold_rho", None),
    VariantSpec("dpmlm_shap", "lambda", "λ-DP-MLM-X+Pr", "token_level_rewriters_with_threshold_lambda", frozenset({"tab"})),
]


def load_csv(dataset: str, suffix: str) -> Optional[pd.DataFrame]:
    path = MDS_DIR / dataset / f"{suffix}.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)


def parse_params(raw: object) -> dict[str, object]:
    if isinstance(raw, dict):
        return raw
    if not isinstance(raw, str):
        return {}
    try:
        result = ast.literal_eval(raw)
        return result if isinstance(result, dict) else {}
    except Exception:
        return {}


def get_mae_references(dataset: str, metric: str) -> tuple[Optional[float], Optional[float]]:
    raw_col = MAE_RAW_COLUMN.get(metric)
    if raw_col is None:
        return None, None
    path = MDS_DIR / dataset / "logs.csv"
    if not path.exists():
        return None, None
    df = pd.read_csv(path)
    if raw_col not in df.columns:
        return None, None

    def first_numeric(rows: pd.DataFrame) -> Optional[float]:
        values = pd.to_numeric(rows[raw_col], errors="coerce").dropna()
        return float(values.iloc[0]) if not values.empty else None

    return (
        first_numeric(df[df["method"] == "baseline"]),
        first_numeric(df[df["method"] == "dummy"]),
    )


def normalize_mae(value: float, baseline_mae: float, dummy_mae: float) -> float:
    if dummy_mae <= baseline_mae:
        return 0.0
    return float(np.clip((dummy_mae - value) / (dummy_mae - baseline_mae), 0.0, 1.0))


def resolve_value(
    raw: object,
    metric: str,
    baseline_mae: Optional[float],
    dummy_mae: Optional[float],
) -> Optional[float]:
    v = pd.to_numeric(raw, errors="coerce")
    if pd.isna(v):
        return None
    if metric in MAE_METRICS and baseline_mae is not None and dummy_mae is not None:
        return normalize_mae(float(v), baseline_mae, dummy_mae)
    return float(v)


def matches_variant(params: dict, variant: VariantSpec) -> bool:
    if variant.threshold_param not in params:
        return False
    return not any(p in params for p in OTHER_THRESHOLD_PARAMS - {variant.threshold_param})


def collect_variant_points(
    df: pd.DataFrame,
    variant: VariantSpec,
    x_metric: str,
    y_metric: str,
    x_ref: tuple[Optional[float], Optional[float]],
    y_ref: tuple[Optional[float], Optional[float]],
) -> dict[Optional[float], list[tuple[float, float]]]:
    if x_metric not in df.columns or y_metric not in df.columns:
        return {}
    groups: dict[Optional[float], list[tuple[float, float]]] = {}
    for _, row in df.iterrows():
        if row.get("method") != variant.method:
            continue
        params = parse_params(row.get("params", {}))
        if not matches_variant(params, variant):
            continue
        x_val = resolve_value(row.get(x_metric), x_metric, *x_ref)
        y_val = resolve_value(row.get(y_metric), y_metric, *y_ref)
        if x_val is None or y_val is None:
            continue
        epsilon: Optional[float] = float(params["epsilon"]) if "epsilon" in params else None
        groups.setdefault(epsilon, []).append((x_val, y_val))
    return groups


def collect_reference_point(
    df: pd.DataFrame,
    method: str,
    x_metric: str,
    y_metric: str,
    x_ref: tuple[Optional[float], Optional[float]],
    y_ref: tuple[Optional[float], Optional[float]],
    epsilon: Optional[float] = None,
) -> Optional[tuple[float, float]]:
    if x_metric not in df.columns or y_metric not in df.columns:
        return None
    pts: list[tuple[float, float]] = []
    for _, row in df.iterrows():
        if row.get("method") != method:
            continue
        params = parse_params(row.get("params", {}))
        if method != "presidio" and any(p in params for p in OTHER_THRESHOLD_PARAMS):
            continue
        if epsilon is not None and params.get("epsilon") != epsilon:
            continue
        x_val = resolve_value(row.get(x_metric), x_metric, *x_ref)
        y_val = resolve_value(row.get(y_metric), y_metric, *y_ref)
        if x_val is None or y_val is None:
            continue
        pts.append((x_val, y_val))
    if not pts:
        return None
    return float(np.mean([p[0] for p in pts])), float(np.mean([p[1] for p in pts]))


def trapezoidal_auc(points: list[tuple[float, float]]) -> float:
    if len(points) < 2:
        return float("nan")
    sorted_pts = sorted(points, key=lambda p: p[0])
    return float(np.trapz([p[1] for p in sorted_pts], [p[0] for p in sorted_pts]))


def _y_references(
    dataset: str,
    y_metric: str,
    y_ref: tuple[Optional[float], Optional[float]],
) -> tuple[Optional[float], Optional[float]]:
    if y_metric not in {"U_nominal_acc", "U_ordinal_mae", "SS_nominal_acc", "SS_ordinal_mae"}:
        return None, None
    if y_metric in MAE_METRICS:
        if y_ref[0] is not None and y_ref[1] is not None:
            return 1.0, 0.0
        return None, None
    raw_col: Optional[str] = {
        "U_nominal_acc": "utility_utility_nominal_raw_acc",
        "SS_nominal_acc": "supervised_similarity_utility_nominal_raw_acc",
    }.get(y_metric)
    if raw_col is None:
        return None, None
    path = MDS_DIR / dataset / "logs.csv"
    if not path.exists():
        return None, None
    df = pd.read_csv(path)
    if raw_col not in df.columns:
        return None, None

    def first_numeric(rows: pd.DataFrame) -> Optional[float]:
        values = pd.to_numeric(rows[raw_col], errors="coerce").dropna()
        return float(values.iloc[0]) if not values.empty else None

    return first_numeric(df[df["method"] == "baseline"]), first_numeric(df[df["method"] == "dummy"])


def plot_showdown(
    dataset: str,
    x_metric: str,
    y_metric: str,
    x_ref: tuple[Optional[float], Optional[float]],
    y_ref: tuple[Optional[float], Optional[float]],
) -> Path:
    fig, axes = create_panel_axes(6, sharex=True, sharey=True, grid_size=(7.5, 6.0))

    masking_curves: list[tuple[str, list[tuple[float, float]]]] = []
    for variant in MASKING_VARIANTS:
        if variant.datasets is not None and dataset not in variant.datasets:
            continue
        df = load_csv(dataset, variant.csv_suffix)
        if df is None:
            continue
        groups = collect_variant_points(df, variant, x_metric, y_metric, x_ref, y_ref)
        pts = groups.get(None, [])
        if pts:
            masking_curves.append((variant.label, pts))

    dp_curves: list[tuple[str, dict[float, list[tuple[float, float]]]]] = []
    for variant in DP_VARIANTS:
        if variant.datasets is not None and dataset not in variant.datasets:
            continue
        df = load_csv(dataset, variant.csv_suffix)
        if df is None:
            continue
        groups = collect_variant_points(df, variant, x_metric, y_metric, x_ref, y_ref)
        eps_groups = {k: v for k, v in groups.items() if k is not None}
        if eps_groups:
            dp_curves.append((variant.label, eps_groups))

    presidio_ref = load_csv(dataset, "risk_maskers")
    presidio_pt: Optional[tuple[float, float]] = None
    if presidio_ref is not None:
        presidio_pt = collect_reference_point(presidio_ref, "presidio", x_metric, y_metric, x_ref, y_ref)

    baseline_y, dummy_y = _y_references(dataset, y_metric, y_ref)

    all_handles: list = []
    all_labels: list[str] = []
    seen_labels: set[str] = set()

    def _add_to_legend(h: object, l: str) -> None:
        if l not in seen_labels:
            seen_labels.add(l)
            all_handles.append(h)
            all_labels.append(l)

    for ax, epsilon_val in zip(axes, FIXED_EPSILON_PANELS):
        is_masking = epsilon_val is None

        if is_masking:
            ax.set_facecolor("#F5F5F5")
            ax.set_title("ε = 0  (masking)", fontsize=11)
            for idx, (label, pts) in enumerate(masking_curves):
                color = _PALETTE[idx % len(_PALETTE)]
                sorted_pts = sorted(pts, key=lambda p: p[0])
                auc = trapezoidal_auc(sorted_pts)
                auc_str = f"{auc:.3f}" if not np.isnan(auc) else "n/a"
                lbl = f"{label}  AUC={auc_str}"
                h, = ax.plot([p[0] for p in sorted_pts], [p[1] for p in sorted_pts],
                             color=color, linewidth=2.0, zorder=4, label=lbl)
                ax.scatter([p[0] for p in sorted_pts], [p[1] for p in sorted_pts],
                           color=color, s=35, zorder=5)
                _add_to_legend(h, lbl)
            if presidio_pt is not None:
                h = ax.scatter(*presidio_pt, marker="*", s=240, color=PRESIDIO_COLOR, zorder=7, label="Presidio")
                _add_to_legend(h, "Presidio")
        else:
            ax.set_title(f"ε = {int(epsilon_val)}", fontsize=11)
            for idx, (label, eps_groups) in enumerate(dp_curves):
                color = _PALETTE[idx % len(_PALETTE)]
                pts = eps_groups.get(epsilon_val, [])
                if len(pts) < 2:
                    continue
                sorted_pts = sorted(pts, key=lambda p: p[0])
                auc = trapezoidal_auc(sorted_pts)
                auc_str = f"{auc:.3f}" if not np.isnan(auc) else "n/a"
                lbl = f"{label}  AUC={auc_str}"
                h, = ax.plot([p[0] for p in sorted_pts], [p[1] for p in sorted_pts],
                             color=color, linewidth=2.0, zorder=4, label=lbl)
                ax.scatter([p[0] for p in sorted_pts], [p[1] for p in sorted_pts],
                           color=color, s=35, zorder=5)
                _add_to_legend(h, label)

        if baseline_y is not None:
            h = ax.axhline(baseline_y, color="#2ca02c", linestyle="--", linewidth=1.3, alpha=0.9, label="original")
            _add_to_legend(h, "original")
        if dummy_y is not None:
            h = ax.axhline(dummy_y, color="#d62728", linestyle="-.", linewidth=1.3, alpha=0.85, label="dummy")
            _add_to_legend(h, "dummy")

        ax.set_xlabel(METRIC_AXIS_LABELS.get(x_metric, x_metric), fontsize=10)
        ax.set_ylabel(METRIC_AXIS_LABELS.get(y_metric, y_metric), fontsize=10)
        ax.grid(True, alpha=0.25)

    if all_handles:
        fig.legend(
            all_handles, all_labels,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.0),
            ncol=min(len(all_handles), 4),
            fontsize=11,
            framealpha=0.9,
            borderaxespad=0.2,
        )
        fig.tight_layout(rect=[0.02, 0.07, 0.98, 0.96])
    else:
        fig.tight_layout(rect=[0.02, 0.02, 0.98, 0.96])

    fig.suptitle(f"{DATASET_LABELS.get(dataset, dataset)} — {x_metric} vs {y_metric}", fontsize=12, y=0.99)
    out_path = OUTPUT_DIR / dataset / x_metric / f"{y_metric}.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def prompt_choice(prompt: str, options: list[str]) -> str:
    print(prompt)
    for i, opt in enumerate(options, 1):
        print(f"  ({i}) {opt}")
    while True:
        raw = input("enter number: ").strip()
        if raw.isdigit() and 1 <= int(raw) <= len(options):
            return options[int(raw) - 1]
        print("invalid choice")


def resolve_pairs(args: argparse.Namespace) -> list[tuple[str, str]]:
    if args.all:
        return [(x, y) for x in X_AXIS_OPTIONS for y in Y_AXIS_OPTIONS]
    x_var: str = args.x_var if args.x_var in METRICS else prompt_choice("select x-axis metric:", METRICS)
    y_var: str = args.y_var if args.y_var in METRICS else prompt_choice("select y-axis metric:", METRICS)
    return [(x_var, y_var)]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Privacy-utility summary plot with AUC for thresholdable methods.",
        epilog=f"Available metrics: {', '.join(METRICS)}",
    )
    parser.add_argument("x_var", nargs="?", help="x-axis metric")
    parser.add_argument("y_var", nargs="?", help="y-axis metric")
    parser.add_argument("--all", action="store_true", help=f"run all {len(X_AXIS_OPTIONS)}×{len(Y_AXIS_OPTIONS)} combinations")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    pairs = resolve_pairs(args)
    plan = [(x_var, y_var, dataset) for x_var, y_var in pairs for dataset in DATASET_LABELS]
    progress = new_progress(total=len(plan), desc="plot_summary", unit="plot")
    saved_count = 0

    for x_var, y_var, dataset in plan:
        progress.set_postfix_str(f"{dataset}:{x_var}->{y_var}")
        x_ref = get_mae_references(dataset, x_var) if x_var in MAE_METRICS else (None, None)
        y_ref = get_mae_references(dataset, y_var) if y_var in MAE_METRICS else (None, None)
        plot_showdown(dataset, x_var, y_var, x_ref, y_ref)
        saved_count += 1
        progress.update(1)

    progress.close()
    print(f"plot_summary: saved={saved_count}, total={len(plan)}")


if __name__ == "__main__":
    main()
