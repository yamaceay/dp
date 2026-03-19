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
VANILLA_COLOR = "#555555"
OTHER_THRESHOLD_PARAMS: frozenset[str] = frozenset({"k", "rho", "lambda"})

METRICS: list[str] = [
    "P_exact", "P_more", "P_full",
    "TRIR_exact", "TRIR_more", "TRIR_full",
    "U_nominal_acc", "U_ordinal_mae",
    "SD_nominal_acc", "SD_ordinal_mae",
    "D_bertscore", "D_cosine", "D_pp",
    "T_anon_avg_s",
]

X_AXIS_OPTIONS: list[str] = ["P_exact", "P_more", "P_full"]
Y_AXIS_OPTIONS: list[str] = ["U_nominal_acc", "U_ordinal_mae", "SD_nominal_acc", "SD_ordinal_mae"]

MAE_METRICS: frozenset[str] = frozenset({"U_ordinal_mae", "SD_ordinal_mae"})

MAE_RAW_COLUMN: dict[str, str] = {
    "U_ordinal_mae": "utility_utility_ordinal_raw_mae",
    "SD_ordinal_mae": "supervised_divergence_utility_ordinal_raw_mae",
}

METRIC_AXIS_LABELS: dict[str, str] = {
    "P_exact": "P_exact (MRR ↓ better)",
    "P_more": "P_more (MRR ↓ better)",
    "P_full": "P_full (MRR ↓ better)",
    "TRIR_exact": "TRIR_exact (↓ better)",
    "TRIR_more": "TRIR_more (↓ better)",
    "TRIR_full": "TRIR_full (↓ better)",
    "U_nominal_acc": "U_nominal (Acc ↑ better)",
    "U_ordinal_mae": "U_ordinal (MAE norm.  [0 = dummy,  1 = original])",
    "SD_nominal_acc": "SD_nominal (Acc ↑ better)",
    "SD_ordinal_mae": "SD_ordinal (MAE norm.  [0 = dummy,  1 = original])",
    "D_bertscore": "D_bertscore",
    "D_cosine": "D_cosine",
    "D_pp": "D_pp",
    "T_anon_avg_s": "Avg. anon time (s)",
}

DATASET_LABELS: dict[str, str] = {"db_bio": "DB-Bio", "tab": "TAB"}


@dataclass(frozen=True)
class VariantSpec:
    method: str
    threshold_param: str
    label: str
    csv_suffix: str
    datasets: Optional[frozenset[str]]


VARIANTS: list[VariantSpec] = [
    VariantSpec("risk_shap", "rho", "Risk+Pr (ρ)", "risk_maskers", None),
    VariantSpec("petre_shap", "k", "PETRE+Pr (k)", "simple_maskers", None),
    VariantSpec("baroud", "lambda", "IPI (λ)", "simple_maskers", frozenset({"tab"})),
    VariantSpec("dpmlm_shap", "k", "k-DP-MLM-X+Pr", "token_level_rewriters_with_threshold_k", None),
    VariantSpec("dpmlm_shap", "rho", "ρ-DP-MLM-X+Pr", "token_level_rewriters_with_threshold_rho", None),
    VariantSpec("dpmlm_shap", "lambda", "λ-DP-MLM-X+Pr", "token_level_rewriters_with_threshold_lambda", frozenset({"tab"})),
]


def load_csv(dataset: str, suffix: str) -> Optional[pd.DataFrame]:
    path = MDS_DIR / f"{dataset}_logs_{suffix}.csv"
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
    path = MDS_DIR / f"{dataset}_logs.csv"
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
        epsilon: Optional[float] = params.get("epsilon")
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


def mean_curve_across_epsilon(
    df: pd.DataFrame,
    variant: VariantSpec,
    x_metric: str,
    y_metric: str,
    x_ref: tuple[Optional[float], Optional[float]],
    y_ref: tuple[Optional[float], Optional[float]],
) -> list[tuple[float, float]]:
    by_threshold: dict[float, list[tuple[float, float]]] = {}
    for _, row in df.iterrows():
        if row.get("method") != variant.method:
            continue
        params = parse_params(row.get("params", {}))
        if not matches_variant(params, variant):
            continue
        threshold_val = float(params[variant.threshold_param])
        x_val = resolve_value(row.get(x_metric), x_metric, *x_ref)
        y_val = resolve_value(row.get(y_metric), y_metric, *y_ref)
        if x_val is None or y_val is None:
            continue
        by_threshold.setdefault(threshold_val, []).append((x_val, y_val))
    return [
        (float(np.mean([p[0] for p in pts])), float(np.mean([p[1] for p in pts])))
        for pts in by_threshold.values()
    ]


def plot_showdown(
    dataset: str,
    x_metric: str,
    y_metric: str,
    x_ref: tuple[Optional[float], Optional[float]],
    y_ref: tuple[Optional[float], Optional[float]],
) -> None:
    fig, ax = plt.subplots(figsize=(8.0, 6.5))

    ref_df = load_csv(dataset, "risk_maskers")
    if ref_df is not None:
        pt = collect_reference_point(ref_df, "presidio", x_metric, y_metric, x_ref, y_ref)
        if pt is not None:
            ax.scatter(*pt, marker="*", s=240, color=PRESIDIO_COLOR, zorder=7, label="Presidio (ref.)")

    vanilla_df = load_csv(dataset, "token_level_rewriters")
    if vanilla_df is not None and x_metric in vanilla_df.columns and y_metric in vanilla_df.columns:
        epsilons = sorted({
            float(parse_params(row.get("params", {})).get("epsilon", 0))
            for _, row in vanilla_df.iterrows()
            if row.get("method") == "dpmlm_shap"
            and not any(p in parse_params(row.get("params", {})) for p in OTHER_THRESHOLD_PARAMS)
        })
        first = True
        for eps in epsilons:
            pt = collect_reference_point(vanilla_df, "dpmlm_shap", x_metric, y_metric, x_ref, y_ref, epsilon=eps)
            if pt is None:
                continue
            ax.scatter(*pt, marker="s", s=50, color=VANILLA_COLOR, alpha=0.65, zorder=6,
                       label="DP-MLM-X+Pr (ref.)" if first else "_nolegend_")
            first = False

    applicable = [v for v in VARIANTS if v.datasets is None or dataset in v.datasets]
    for idx, variant in enumerate(applicable):
        color = _PALETTE[idx % len(_PALETTE)]
        df = load_csv(dataset, variant.csv_suffix)
        if df is None:
            continue

        epsilon_groups = collect_variant_points(df, variant, x_metric, y_metric, x_ref, y_ref)
        if not epsilon_groups:
            continue

        per_eps_aucs: list[float] = []
        for eps, pts in sorted(epsilon_groups.items(), key=lambda x: (x[0] is None, x[0] or 0)):
            if len(pts) < 2:
                continue
            sorted_pts = sorted(pts, key=lambda p: p[0])
            ax.plot([p[0] for p in sorted_pts], [p[1] for p in sorted_pts], color=color, alpha=0.18, linewidth=0.8)
            ax.scatter([p[0] for p in sorted_pts], [p[1] for p in sorted_pts], color=color, alpha=0.3, s=10, zorder=3)
            auc = trapezoidal_auc(pts)
            if not np.isnan(auc):
                per_eps_aucs.append(auc)

        mean_pts = mean_curve_across_epsilon(df, variant, x_metric, y_metric, x_ref, y_ref)
        if len(mean_pts) < 2:
            if per_eps_aucs:
                ax.plot([], [], color=color, linewidth=2.0, label=f"{variant.label}  AUC={np.mean(per_eps_aucs):.3f}")
            continue

        mean_sorted = sorted(mean_pts, key=lambda p: p[0])
        mean_auc = float(np.mean(per_eps_aucs)) if per_eps_aucs else trapezoidal_auc(mean_pts)
        auc_str = f"{mean_auc:.3f}" if not np.isnan(mean_auc) else "n/a"
        ax.plot([p[0] for p in mean_sorted], [p[1] for p in mean_sorted],
                color=color, linewidth=2.0, zorder=4, label=f"{variant.label}  AUC={auc_str}")
        ax.scatter([p[0] for p in mean_sorted], [p[1] for p in mean_sorted], color=color, s=35, zorder=5)

    ax.set_xlabel(METRIC_AXIS_LABELS.get(x_metric, x_metric), fontsize=11)
    ax.set_ylabel(METRIC_AXIS_LABELS.get(y_metric, y_metric), fontsize=11)
    ax.set_title(f"{DATASET_LABELS.get(dataset, dataset)} — {x_metric} vs {y_metric}", fontsize=12)
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles, labels, fontsize=8.5, loc="best", framealpha=0.9)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()

    out_path = OUTPUT_DIR / f"{dataset}_{x_metric}_vs_{y_metric}.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"saved: {out_path}")


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

    for x_var, y_var in resolve_pairs(args):
        for dataset in DATASET_LABELS:
            x_ref = get_mae_references(dataset, x_var) if x_var in MAE_METRICS else (None, None)
            y_ref = get_mae_references(dataset, y_var) if y_var in MAE_METRICS else (None, None)
            plot_showdown(dataset, x_var, y_var, x_ref, y_ref)


if __name__ == "__main__":
    main()
