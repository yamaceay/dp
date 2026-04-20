from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd
import yaml

MDS_DIR = Path("mds")
DATASET_CONFIG_PATH = Path("configs/visualize/datasets.yaml")

MethodFilter = Callable[[str, dict], bool]


@dataclass(frozen=True)
class Variant:
    group: str
    label: str
    color: str
    filter: MethodFilter


def _only(params: dict, *keys: str) -> bool:
    return set(params.keys()) == set(keys)


def _has(params: dict, *keys: str) -> bool:
    return all(k in params for k in keys)


VARIANTS: list[Variant] = [
    Variant("Reference",     "Original",             "#2ca02c", lambda m, _: m == "baseline"),
    Variant("Reference",     "Dummy",                "#d62728", lambda m, _: m == "dummy"),
    Variant("Reference",     "Presidio",             "#8F2D56", lambda m, _: m == "presidio"),
    Variant("Reference",     "SpaCy",                "#7f7f7f", lambda m, _: m == "spacy"),
    Variant("Reference",     "Manual",               "#aaaaaa", lambda m, _: m == "manual"),
    Variant("Token masking", "PETRE+Pr  (k)",         "#F0A500", lambda m, p: m == "petre_shap" and _only(p, "k")),
    Variant("Token masking", "Risk+Pr  (ρ)",          "#C17800", lambda m, p: m == "risk_shap" and _only(p, "rho")),
    Variant("Token masking", "IPI+Pr  (λ)",           "#8B5E00", lambda m, p: m == "baroud" and _only(p, "lambda")),
    Variant("Sentence DP",   "DP-BART  (ε)",         "#D55E00", lambda m, p: m == "dpbart" and _only(p, "epsilon")),
    Variant("Sentence DP",   "DP-Paraphrase  (ε)",   "#E69F00", lambda m, p: m == "dpparaphrase" and _only(p, "epsilon")),
    Variant("Sentence DP",   "DP-Prompt  (ε)",       "#CC79A7", lambda m, p: m == "dpprompt" and _only(p, "epsilon")),
    Variant("Token DP",      "DP-MLM  (ε)",          "#56B4E9", lambda m, p: m == "dpmlm_uniform" and _only(p, "epsilon")),
    Variant("Token DP",      "DP-MLM-X  (ε)",        "#0072B2", lambda m, p: m == "dpmlm_shap_no_presidio" and _only(p, "epsilon")),
    Variant("Token DP",      "DP-MLM-X+Pr  (ε)",     "#003f7f", lambda m, p: m == "dpmlm_shap" and _only(p, "epsilon")),
    Variant("Token DP",      "DP-MLM-X+Pr  (ε, k)",  "#009E73", lambda m, p: m == "dpmlm_shap" and _has(p, "epsilon", "k") and "rho" not in p and "lambda" not in p),
    Variant("Token DP",      "DP-MLM-X+Pr  (ε, ρ)",  "#005C42", lambda m, p: m == "dpmlm_shap" and _has(p, "epsilon", "rho") and "k" not in p and "lambda" not in p),
    Variant("Token DP",      "DP-MLM-X+Pr  (ε, λ)",  "#2D6A4F", lambda m, p: m == "dpmlm_shap" and _has(p, "epsilon", "lambda") and "k" not in p and "rho" not in p),
]

GROUP_ORDER: list[str] = ["Reference", "Token masking", "Sentence DP", "Token DP"]

_PP_NOT_APPLICABLE: frozenset[str] = frozenset({"dpbart", "dpparaphrase", "dpprompt"})

METRIC_METHOD_EXCLUSIONS: dict[str, frozenset[str]] = {
    "divergence_pp": _PP_NOT_APPLICABLE,
}

METRIC_METHOD_FOOTNOTES: dict[str, frozenset[str]] = {}


def parse_params(raw: object) -> dict:
    if isinstance(raw, dict):
        return raw
    try:
        parsed = ast.literal_eval(str(raw))
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass
    return {}


def read_dataset_labels() -> dict[str, str]:
    with open(DATASET_CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f)
    labels: dict[str, str] = {}
    for dataset_set in config.get("dataset_sets", []):
        names = dataset_set.get("names", [])
        for entry in names if isinstance(names, list) else []:
            if isinstance(entry, dict):
                name, pa = entry.get("name"), entry.get("print_as")
                if isinstance(name, str) and isinstance(pa, str):
                    labels[name] = pa
            elif isinstance(entry, str):
                labels.setdefault(entry, entry.replace("_", "-").upper())
        if not names:
            name = dataset_set.get("name")
            if isinstance(name, str):
                labels.setdefault(name, dataset_set.get("print_as", name))
    return labels


def load_meta(dataset: str) -> Optional[pd.DataFrame]:
    path = MDS_DIR / dataset / "meta_logs.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    main = df[df["split"].isna() | (df["split"] == "")].copy()
    main["params_dict"] = main["params"].apply(parse_params)
    return main


@dataclass
class VariantStats:
    variant: Variant
    entries: list[tuple[dict, float]]  # (params, value) sorted by value desc
    not_applicable: bool = False
    footnote: bool = False

    @property
    def values(self) -> list[float]:
        return [v for _, v in self.entries]

    @property
    def mean(self) -> float:
        return float(np.mean(self.values))

    @property
    def lo(self) -> float:
        return float(np.min(self.values))

    @property
    def hi(self) -> float:
        return float(np.max(self.values))

    @property
    def n(self) -> int:
        return len(self.entries)


def compute_variant_stats(df: pd.DataFrame, col: str) -> list[VariantStats]:
    excluded = METRIC_METHOD_EXCLUSIONS.get(col, frozenset())
    footnoted = METRIC_METHOD_FOOTNOTES.get(col, frozenset())
    stats: list[VariantStats] = []
    for v in VARIANTS:
        mask = df.apply(lambda row: v.filter(str(row["method"]), row["params_dict"]), axis=1)
        if excluded:
            excluded_mask = mask & df["method"].isin(excluded)
            mask = mask & ~df["method"].isin(excluded)
            if excluded_mask.any() and not mask.any():
                stats.append(VariantStats(variant=v, entries=[], not_applicable=True))
                continue
        subset = df.loc[mask].dropna(subset=[col])
        entries = sorted(
            [(row["params_dict"], float(row[col])) for _, row in subset.iterrows()],
            key=lambda t: t[1],
            reverse=True,
        )
        if entries:
            is_footnote = footnoted and df.loc[mask, "method"].isin(footnoted).all()
            stats.append(VariantStats(variant=v, entries=entries, footnote=is_footnote))
    return stats
