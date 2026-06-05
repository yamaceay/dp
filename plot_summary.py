from __future__ import annotations

import ast
import argparse
import csv
import sys
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

from plot_layout import create_panel_axes, REFERENCE_STYLES
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
OTHER_THRESHOLD_PARAMS: frozenset[str] = frozenset({"k", "rho", "lambda"})

METRICS: list[str] = [
    "P_exact", "P_more", "P_full",
    "TRIR_exact", "TRIR_more", "TRIR_full",
    "U_acc", "U_mae",
    "PU_acc", "PU_mae",
    "D_bertscore", "D_cosine", "D_pp",
    "T_anon_avg_s",
]

FIXED_EPSILON_PANELS: list[float | None] = [None, 10, 25, 50, 100, 250]

X_AXIS_OPTIONS: list[str] = ["P_exact", "P_more", "P_full", "TRIR_exact", "TRIR_more", "TRIR_full"]
Y_AXIS_OPTIONS: list[str] = [
    "U_acc", "U_mae",
    "PU_acc", "PU_mae",
    "D_bertscore", "D_cosine", "D_pp",
    "T_anon_avg_s",
]

MAE_METRICS: frozenset[str] = frozenset({"U_ordinal_mae", "PU_ordinal_mae"})

MAE_RAW_COLUMN: dict[str, str] = {
    "U_ordinal_mae": "utility_utility_ordinal_raw_mae",
    "PU_ordinal_mae": "pseudo_utility_utility_ordinal_raw_mae",
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
    "PU_acc": "PU_acc (Acc ↑ better)",
    "PU_mae": "PU_mae (MAE ↓ better)",
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
    dp_panel_ref: bool = True  # show as dashed reference in ε > 0 panels


# Ordered to match DP_VARIANTS (k, rho, lambda) so palette indices align.
MASKING_VARIANTS: list[VariantSpec] = [
    VariantSpec("petre_shap", "k",      "PETRE+Pr (k)",  "token_level_rewriters_with_threshold_k",      None,                  dp_panel_ref=False),
    VariantSpec("risk_shap",  "rho",    "Risk+Pr (ρ)",   "token_level_rewriters_with_threshold_rho",     None,                  dp_panel_ref=False),
    VariantSpec("baroud",     "lambda", "IPI (λ)",       "token_level_rewriters_with_threshold_lambda",  frozenset({"tab"}),    dp_panel_ref=False),
]

# Unconstrained single-point markers for each ε panel (no stopping condition).
UNCONSTRAINED_ENDPOINTS: list[tuple[str, str, str]] = [
    ("dpmlm_uniform", "DP-MLM",      _PALETTE[3]),
    ("dpmlm_shap",    "DP-MLM-X+Pr", _PALETTE[4]),
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
) -> dict[Optional[float], list[tuple[float, float, Optional[float]]]]:
    if x_metric not in df.columns or y_metric not in df.columns:
        return {}
    groups: dict[Optional[float], list[tuple[float, float, Optional[float]]]] = {}
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
        thr_raw = params.get(variant.threshold_param)
        thr: Optional[float] = float(thr_raw) if thr_raw is not None else None
        groups.setdefault(epsilon, []).append((x_val, y_val, thr))
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


def trapezoidal_auc(points: list[tuple]) -> float:
    if len(points) < 2:
        return float("nan")
    return float(np.trapz([p[1] for p in points], [p[0] for p in points]))



def plot_showdown(
    dataset: str,
    x_metric: str,
    y_metric: str,
    x_ref: tuple[Optional[float], Optional[float]],
    y_ref: tuple[Optional[float], Optional[float]],
) -> list[dict]:
    """Returns AUC records for the manifest; saves the plot as a side-effect."""
    fig, axes = create_panel_axes(6, sharex=True, sharey=True, grid_size=(7.5, 6.0))
    auc_records: list[dict] = []

    # (variant_list_idx, label, pts, dp_panel_ref) — idx kept for stable palette colours
    masking_data: list[tuple[int, str, list, bool]] = []
    for idx, variant in enumerate(MASKING_VARIANTS):
        if variant.datasets is not None and dataset not in variant.datasets:
            continue
        df = load_csv(dataset, variant.csv_suffix)
        if df is None:
            continue
        pts = collect_variant_points(df, variant, x_metric, y_metric, x_ref, y_ref).get(None, [])
        if pts:
            masking_data.append((idx, variant.label, pts, variant.dp_panel_ref))

    # (variant_list_idx, label, eps_groups)
    dp_data: list[tuple[int, str, dict]] = []
    for idx, variant in enumerate(DP_VARIANTS):
        if variant.datasets is not None and dataset not in variant.datasets:
            continue
        df = load_csv(dataset, variant.csv_suffix)
        if df is None:
            continue
        groups = collect_variant_points(df, variant, x_metric, y_metric, x_ref, y_ref)
        eps_groups = {k: v for k, v in groups.items() if k is not None}
        if eps_groups:
            dp_data.append((idx, variant.label, eps_groups))

    _smr = load_csv(dataset, "simple_maskers_and_rewriters")
    reference_pts: dict[str, Optional[tuple[float, float]]] = {}
    for _ref_method in REFERENCE_STYLES:
        reference_pts[_ref_method] = (
            collect_reference_point(_smr, _ref_method, x_metric, y_metric, x_ref, y_ref)
            if _smr is not None else None
        )

    all_handles: list = []
    all_labels: list[str] = []
    seen_labels: set[str] = set()

    def _add_to_legend(h: object, l: str) -> None:
        if l not in seen_labels:
            seen_labels.add(l)
            all_handles.append(h)
            all_labels.append(l)

    def _draw_curve(ax, sorted_pts, color, label, *, dashed: bool) -> object:
        ls = "--" if dashed else "-"
        lw = 1.5 if dashed else 2.0
        alpha_sc = 0.7 if dashed else 1.0
        h, = ax.plot([p[0] for p in sorted_pts], [p[1] for p in sorted_pts],
                     color=color, linewidth=lw, linestyle=ls, zorder=4 if not dashed else 3, label=label)
        ax.scatter([p[0] for p in sorted_pts], [p[1] for p in sorted_pts],
                   color=color, s=35 if not dashed else 20, zorder=5 if not dashed else 4, alpha=alpha_sc)
        for x, y, thr in sorted_pts:
            if thr is not None:
                ax.annotate(f"{thr:g}", (x, y), textcoords="offset points",
                            xytext=(3, 3), fontsize=6, color=color, alpha=0.85 if not dashed else 0.7)
        return h

    for ax, epsilon_val in zip(axes, FIXED_EPSILON_PANELS):
        is_masking = epsilon_val is None

        if is_masking:
            ax.set_facecolor("#F5F5F5")
            ax.set_title("ε = 0  (masking)", fontsize=11)
            for p_idx, label, pts, _dp_ref in masking_data:
                color = _PALETTE[p_idx % len(_PALETTE)]
                auc = trapezoidal_auc(pts)
                auc_records.append({"dataset": dataset, "x": x_metric, "y": y_metric,
                                    "variant": label, "epsilon": "det",
                                    "auc": round(auc, 5) if not np.isnan(auc) else None})
                h = _draw_curve(ax, pts, color, label, dashed=True)
                _add_to_legend(h, label)
        else:
            ax.set_title(f"ε = {int(epsilon_val)}", fontsize=11)
            for p_idx, label, eps_groups in dp_data:
                color = _PALETTE[p_idx % len(_PALETTE)]
                pts = eps_groups.get(epsilon_val, [])
                if len(pts) < 2:
                    continue
                auc = trapezoidal_auc(pts)
                auc_records.append({"dataset": dataset, "x": x_metric, "y": y_metric,
                                    "variant": label, "epsilon": epsilon_val,
                                    "auc": round(auc, 5) if not np.isnan(auc) else None})
                h = _draw_curve(ax, pts, color, label, dashed=False)
                _add_to_legend(h, label)
            # Deterministic reference curves (dp_panel_ref only) — dashed
            for p_idx, label, pts, dp_ref in masking_data:
                if not dp_ref or not pts:
                    continue
                color = _PALETTE[p_idx % len(_PALETTE)]
                lbl = f"{label}  (det.)"
                h = _draw_curve(ax, pts, color, lbl, dashed=True)
                _add_to_legend(h, lbl)
            # Unconstrained single-point markers
            if _smr is not None:
                for ep_method, ep_label, ep_color in UNCONSTRAINED_ENDPOINTS:
                    ep_pt = collect_reference_point(
                        _smr, ep_method, x_metric, y_metric, x_ref, y_ref, epsilon=epsilon_val
                    )
                    if ep_pt is None:
                        continue
                    lbl = f"{ep_label} (no stop)"
                    h = ax.scatter(*ep_pt, marker="X", s=90, color=ep_color, zorder=6,
                                   linewidths=0.6, edgecolors="white", label=lbl)
                    _add_to_legend(h, lbl)

        # Reference scatter in ALL panels — grounds the absolute scale
        for _ref, (_lbl, _mkr, _sz, _col) in REFERENCE_STYLES.items():
            _rpt = reference_pts.get(_ref)
            if _rpt is None:
                continue
            h = ax.scatter(*_rpt, marker=_mkr, s=_sz, color=_col, zorder=7, label=_lbl)
            _add_to_legend(h, _lbl)

        # Vertical + horizontal reference lines for baseline and dummy
        for _ref, _vcolor, _vstyle in (
            ("baseline", "#2ca02c", "--"),
            ("dummy",    "#d62728", "-."),
        ):
            _rpt = reference_pts.get(_ref)
            if _rpt is not None:
                ax.axvline(_rpt[0], color=_vcolor, linestyle=_vstyle, linewidth=1.0, alpha=0.4, zorder=2)
                ax.axhline(_rpt[1], color=_vcolor, linestyle=_vstyle, linewidth=1.0, alpha=0.4, zorder=2)

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
    return auc_records


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


META_X_OPTIONS: list[str] = ["P"]
META_Y_OPTIONS: list[str] = ["U", "PU", "D"]


def plot_meta_showdown(dataset: str, x_col: str, y_col: str) -> None:
    from plot_mega_shared import (
        GROUP_ORDER, VARIANTS, compute_variant_stats, load_meta, read_dataset_labels,
    )
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    df = load_meta(dataset)
    if df is None or x_col not in df.columns or y_col not in df.columns:
        return

    x_stats = {s.variant.label: s for s in compute_variant_stats(df, x_col)}
    y_stats = {s.variant.label: s for s in compute_variant_stats(df, y_col)}
    common = [v for v in VARIANTS if v.label in x_stats and v.label in y_stats]

    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    for v in common:
        sx, sy = x_stats[v.label], y_stats[v.label]
        color = v.color
        ax.hlines(sy.mean, sx.lo, sx.hi, colors=color, linewidth=3.0, alpha=0.45, zorder=3)
        ax.vlines(sx.mean, sy.lo, sy.hi, colors=color, linewidth=3.0, alpha=0.45, zorder=3)
        ax.scatter(sx.mean, sy.mean, s=80, color=color, zorder=5,
                   linewidths=0.8, edgecolors="white")
        ax.annotate(v.label.strip(), (sx.mean, sy.mean),
                    textcoords="offset points", xytext=(5, 4),
                    fontsize=6.5, color=color)

    ax.set_xlabel(x_col, fontsize=10)
    ax.set_ylabel(y_col, fontsize=10)
    ax.set_title(f"{DATASET_LABELS.get(dataset, dataset)} — {x_col} vs {y_col}",
                 fontsize=11, fontweight="bold")
    ax.grid(True, alpha=0.25, linewidth=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    out = OUTPUT_DIR / "meta" / dataset / x_col / f"{y_col}.pdf"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Privacy-utility summary plot with AUC for thresholdable methods.",
        epilog=f"Available metrics: {', '.join(METRICS)}",
    )
    parser.add_argument("x_var", nargs="?", help="x-axis metric")
    parser.add_argument("y_var", nargs="?", help="y-axis metric")
    parser.add_argument("--all", action="store_true", help=f"run all {len(X_AXIS_OPTIONS)}×{len(Y_AXIS_OPTIONS)} combinations")
    parser.add_argument("--meta", action="store_true", help="Use meta metrics (P/U/SS/D from meta_logs.csv)")
    parser.add_argument("--dataset", nargs="*", dest="datasets", metavar="DATASET",
                        help="Datasets to include (default: all)")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    selected_datasets = args.datasets if args.datasets else list(DATASET_LABELS)

    if args.meta:
        pairs = [(x, y) for x in META_X_OPTIONS for y in META_Y_OPTIONS]
        plan = [(x, y, ds) for x, y in pairs for ds in selected_datasets if ds in DATASET_LABELS]
        progress = new_progress(total=len(plan), desc="plot_summary --meta", unit="plot")
        for x_var, y_var, dataset in plan:
            progress.set_postfix_str(f"{dataset}:{x_var}->{y_var}")
            plot_meta_showdown(dataset, x_var, y_var)
            progress.update(1)
        progress.close()
        return

    pairs = resolve_pairs(args)
    plan = [(x_var, y_var, ds) for x_var, y_var in pairs for ds in selected_datasets if ds in DATASET_LABELS]
    progress = new_progress(total=len(plan), desc="plot_summary", unit="plot")
    saved_count = 0
    all_auc_records: list[dict] = []

    for x_var, y_var, dataset in plan:
        progress.set_postfix_str(f"{dataset}:{x_var}->{y_var}")
        x_ref = get_mae_references(dataset, x_var) if x_var in MAE_METRICS else (None, None)
        y_ref = get_mae_references(dataset, y_var) if y_var in MAE_METRICS else (None, None)
        try:
            records = plot_showdown(dataset, x_var, y_var, x_ref, y_ref)
            all_auc_records.extend(records)
            saved_count += 1
        except Exception as exc:
            print(f"\n[warn] {dataset}:{x_var}->{y_var} failed: {exc}", file=sys.stderr)
        progress.update(1)

    progress.close()

    if all_auc_records:
        manifest_path = OUTPUT_DIR / "auc_manifest.csv"
        with open(manifest_path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=["dataset", "x", "y", "variant", "epsilon", "auc"])
            writer.writeheader()
            writer.writerows(all_auc_records)
        print(f"plot_summary: AUC manifest → {manifest_path}")

    print(f"plot_summary: saved={saved_count}, total={len(plan)}")


if __name__ == "__main__":
    main()
