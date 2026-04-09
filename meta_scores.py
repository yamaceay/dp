from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd
import yaml

_ORIGINAL_RE = re.compile(r"^original\((\w+)\)$")


@dataclass(frozen=True)
class MetaScoreDef:
    output: str
    op: str
    args: tuple[str, ...]
    split: Optional[str]
    best: Optional[float]
    worst: Optional[float]
    category: str
    strategy: str  # "exact_set" | "smallest_subset"


def _parse_bound(v: object) -> Optional[float]:
    if v is None:
        return None
    if isinstance(v, (int, float)):
        return float(v)
    s = str(v).strip().lower()
    if s in ('float("inf")', "float('inf')", "inf"):
        return math.inf
    if s in ('float("-inf")', "float('-inf')", "-inf"):
        return -math.inf
    try:
        return float(s)
    except ValueError:
        return None


def _derive_bounds(
    op: str,
    operand_bounds: list[tuple[float, float]],
) -> tuple[Optional[float], Optional[float]]:
    """
    Propagate (best, worst) through an operation using the rule:
    "the result is at its best when every operand is at its best."

    avg: best = mean(best_i), worst = mean(worst_i)
    sub: best = best_a - best_b, worst = worst_a - worst_b
    div: best = best_a / best_b, worst = worst_a / worst_b
         returns (None, None) when any denominator bound is zero.
    """
    if op == "avg":
        bests = [b for b, _ in operand_bounds]
        worsts = [w for _, w in operand_bounds]
        return sum(bests) / len(bests), sum(worsts) / len(worsts)
    if op == "sub":
        (ba, wa), (bb, wb) = operand_bounds
        return ba - bb, wa - wb
    if op == "div":
        (ba, wa), (bb, wb) = operand_bounds
        if bb == 0.0 or wb == 0.0:
            return None, None
        return ba / bb, wa / wb
    return None, None


def _resolve_bounds(
    defs: list[MetaScoreDef],
    seed_bounds: dict[str, tuple[float, float]],
) -> list[MetaScoreDef]:
    """
    For each def without explicit best/worst, derive them from already-known bounds.
    Explicit YAML bounds always take priority.
    """
    known: dict[str, tuple[float, float]] = dict(seed_bounds)

    resolved: list[MetaScoreDef] = []
    for d in defs:
        if d.best is not None and d.worst is not None:
            resolved.append(d)
            known.setdefault(d.output, (d.best, d.worst))
            continue

        operand_bounds: list[tuple[float, float]] = []
        all_known = True
        for arg in d.args:
            m = _ORIGINAL_RE.match(arg)
            name = m.group(1) if m else arg
            if name not in known:
                all_known = False
                break
            operand_bounds.append(known[name])

        if all_known:
            db, dw = _derive_bounds(d.op, operand_bounds)
        else:
            db, dw = None, None

        best = d.best if d.best is not None else db
        worst = d.worst if d.worst is not None else dw
        new_d = MetaScoreDef(
            output=d.output,
            op=d.op,
            args=d.args,
            split=d.split,
            best=best,
            worst=worst,
            category=d.category,
            strategy=d.strategy,
        )
        resolved.append(new_d)
        if best is not None and worst is not None:
            known.setdefault(d.output, (best, worst))

    return resolved


def parse_meta_score_defs(config_path: Path) -> list[MetaScoreDef]:
    with config_path.open("r") as f:
        config = yaml.safe_load(f)

    seed_bounds: dict[str, tuple[float, float]] = {}
    for dataset_set in config.get("dataset_sets", []):
        if not isinstance(dataset_set, dict):
            continue
        for score in dataset_set.get("scores", []):
            if not isinstance(score, dict):
                continue
            name = score.get("rename_as") or score.get("name")
            best = _parse_bound(score.get("best"))
            worst = _parse_bound(score.get("worst"))
            if isinstance(name, str) and best is not None and worst is not None:
                seed_bounds[name] = (best, worst)

    raw_defs: list[MetaScoreDef] = []
    for dataset_set in config.get("dataset_sets", []):
        if not isinstance(dataset_set, dict):
            continue
        for raw in dataset_set.get("meta_scores", []):
            if not isinstance(raw, dict):
                continue
            output = raw.get("rename_as")
            if not isinstance(output, str):
                continue
            split_val = raw.get("split")
            split: Optional[str] = str(split_val).strip() if split_val is not None else None
            if split == "":
                split = None
            if "avg" in raw:
                op, args = "avg", tuple(str(a) for a in raw["avg"])
            elif "div" in raw:
                op, args = "div", tuple(str(a) for a in raw["div"])
            elif "sub" in raw:
                op, args = "sub", tuple(str(a) for a in raw["sub"])
            else:
                continue
            raw_defs.append(MetaScoreDef(
                output=output,
                op=op,
                args=args,
                split=split,
                best=_parse_bound(raw.get("best")),
                worst=_parse_bound(raw.get("worst")),
                category=str(raw.get("category", "meta")),
                strategy=str(raw.get("strategy", "exact_set")),
            ))

    return _resolve_bounds(raw_defs, seed_bounds)


def _row_split(row: pd.Series) -> str:
    val = row.get("split")
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return ""
    return str(val)


def _select_def(defs: list[MetaScoreDef], split: str) -> Optional[MetaScoreDef]:
    for d in defs:
        if d.split is not None and d.split == split:
            return d
    for d in defs:
        if d.split is None:
            return d
    return None


def _to_float(v: object) -> float:
    if v is None:
        return math.nan
    try:
        f = float(v)  # type: ignore[arg-type]
        return f if math.isfinite(f) else math.nan
    except (TypeError, ValueError):
        return math.nan


def _lookup_baseline(col: str, split: str, df: pd.DataFrame) -> float:
    mask = (df["method"] == "baseline") & (df["split"].fillna("").astype(str) == split)
    rows = df[mask]
    if rows.empty:
        return math.nan
    return _to_float(rows.iloc[0].get(col))


def _apply_op(op: str, operands: list[float]) -> float:
    if op == "avg":
        return sum(operands) / len(operands)
    if op == "div":
        if operands[1] == 0.0:
            return math.nan
        return operands[0] / operands[1]
    if op == "sub":
        return operands[0] - operands[1]
    raise ValueError(f"Unknown op: {op!r}")


def _applicable_rows(
    df: pd.DataFrame,
    split: Optional[str],
    other_splits: list[str],
) -> pd.DataFrame:
    """
    Rows this def applies to.
    - split-specific def: rows where split column matches.
    - fallback def (split=None): rows NOT covered by any split-specific sibling.
    """
    col = df["split"].fillna("").astype(str)
    if split is not None:
        return df[col == split]
    if other_splits:
        return df[~col.isin(other_splits)]
    return df


def _find_common_subset(args: tuple[str, ...], rows: pd.DataFrame) -> list[str]:
    """
    Among rows that have at least one non-NaN arg, return the args that are
    non-NaN for ALL such rows (intersection).
    """
    if rows.empty:
        return []

    present_cols = [a for a in args if a in rows.columns]
    if not present_cols:
        return []

    has_any = rows[present_cols].apply(
        lambda col: col.map(lambda v: not math.isnan(_to_float(v)))
    ).any(axis=1)
    relevant = rows[has_any]
    if relevant.empty:
        return []

    common: list[str] = []
    for arg in args:
        if arg not in rows.columns:
            continue
        all_present = relevant[arg].apply(lambda v: not math.isnan(_to_float(v))).all()
        if all_present:
            common.append(arg)
    return common


def _validate_exact_set(
    args: tuple[str, ...],
    output: str,
    rows: pd.DataFrame,
) -> None:
    """
    For rows that have at least one non-NaN arg, all args must be non-NaN.
    Raises ValueError on first violation.
    """
    for _, row in rows.iterrows():
        values = {a: _to_float(row.get(a)) for a in args}
        any_present = any(not math.isnan(v) for v in values.values())
        if not any_present:
            continue
        missing = [a for a, v in values.items() if math.isnan(v)]
        if missing:
            method = row.get("method", "?")
            raise ValueError(
                f"exact_set violated for '{output}': method={method!r} has "
                f"[{', '.join(missing)}] missing while other args are present."
            )


class MetaScoreEngine:
    def __init__(self, defs: list[MetaScoreDef]) -> None:
        self._defs = defs
        self.resolved_args: dict[str, list[str]] = {}

    def meta_columns(self) -> list[str]:
        seen: list[str] = []
        for d in self._defs:
            if d.output not in seen:
                seen.append(d.output)
        return seen

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        result = df.copy()
        seen: list[str] = []
        grouped: dict[str, list[MetaScoreDef]] = {}
        for d in self._defs:
            if d.output not in grouped:
                seen.append(d.output)
                grouped[d.output] = []
            grouped[d.output].append(d)

        self.resolved_args = {}
        effective: dict[tuple[str, Optional[str]], tuple[str, ...]] = {}

        for output in seen:
            defs_for_output = grouped[output]
            specific_splits = [d.split for d in defs_for_output if d.split is not None]

            for d in defs_for_output:
                if d.op != "avg":
                    effective[(output, d.split)] = d.args
                    continue

                rows = _applicable_rows(result, d.split, specific_splits)

                if d.strategy == "smallest_subset":
                    common = _find_common_subset(d.args, rows)
                    effective[(output, d.split)] = tuple(common)
                elif d.strategy == "exact_set":
                    _validate_exact_set(d.args, output, rows)
                    effective[(output, d.split)] = d.args
                else:
                    raise ValueError(f"Unknown strategy {d.strategy!r} for '{output}'.")

            combined = set()
            for d in defs_for_output:
                combined.update(effective.get((output, d.split), d.args))
            self.resolved_args[output] = sorted(combined)

        for output in seen:
            defs_for_output = grouped[output]
            col_values: list[float] = []
            for _, row in result.iterrows():
                split = _row_split(row)
                d = _select_def(defs_for_output, split)
                if d is None:
                    col_values.append(math.nan)
                    continue
                active = effective.get((output, d.split), d.args)
                if not active:
                    col_values.append(math.nan)
                    continue
                operands: list[float] = []
                for arg in active:
                    m = _ORIGINAL_RE.match(arg)
                    if m:
                        v = _lookup_baseline(m.group(1), split, result)
                    else:
                        v = _to_float(row.get(arg))
                    operands.append(v)
                if any(math.isnan(v) for v in operands):
                    col_values.append(math.nan)
                else:
                    col_values.append(_apply_op(d.op, operands))
            result[output] = col_values

        return result
