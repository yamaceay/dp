"""
Evaluates RAT-Bench SHAP risk scores against gold direct-identifier spans:
for each record, tokens overlapping a locatable direct identifier (name,
email, phone number, SSN, address, credit card number) are the positive
class; per-record ROC-AUC then asks whether higher risk scores rank those
tokens above the rest of the record. Indirect identifiers (ACS codes such as
ESR, RAC2P) are excluded -- they essentially never appear verbatim in the
text, so they have no usable gold offsets (see dp/loaders/_ratbench.py).

Aggregates per model (bart-seeds vs. nobart-seeds, same 12 models as
risk_drift_diff_attackers.py's groups mode) and runs an exact group-label
permutation test on the per-model mean-AUC difference between regimes.
"""

import argparse
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from dp.loaders import get_adapter
from risk_drift_diff_attackers import (
    _assignment_count,
    _label_symmetry_factor,
    _multiset_assignments,
    load_shap,
)

DATA_DIR = Path("data")


def load_gold_direct_spans(dataset: str, data_in: str) -> dict[str, list[tuple[int, int]]]:
    adapter = get_adapter(dataset, data=dataset, data_in=data_in)
    gold: dict[str, list[tuple[int, int]]] = {}
    for record in adapter.iter_records():
        spans = [
            (span.start, span.end)
            for span in (record.spans or [])
            if span.metadata.get("category") == "direct" and span.start >= 0 and span.end > span.start
        ]
        gold[str(record.uid)] = spans
    return gold


def _overlaps(token: tuple[int, int], spans: list[tuple[int, int]]) -> bool:
    t0, t1 = token
    return any(t0 < s1 and t1 > s0 for s0, s1 in spans)


def per_record_gold_auc(
    shap: dict[str, dict],
    gold: dict[str, list[tuple[int, int]]],
) -> tuple[np.ndarray, int, int]:
    """Per-record ROC-AUC of risk score vs. "is this token part of a
    locatable direct identifier". Records with no locatable gold span, or
    where every token is (or isn't) inside one, have an undefined AUC and are
    skipped rather than silently scored.
    """
    aucs: list[float] = []
    skipped_no_gold = 0
    skipped_degenerate = 0
    for uid, rec in shap.items():
        spans = gold.get(uid, [])
        if not spans:
            skipped_no_gold += 1
            continue
        labels = np.array([1 if _overlaps(tuple(o), spans) else 0 for o in rec["offsets"]])
        if labels.sum() == 0 or labels.sum() == len(labels):
            skipped_degenerate += 1
            continue
        scores = np.asarray(rec["scores"], dtype=float)
        aucs.append(float(roc_auc_score(labels, scores)))
    return np.asarray(aucs, dtype=float), skipped_no_gold, skipped_degenerate


def group_mean_permutation_test(
    values: dict[str, float],
    group_of: dict[str, str],
) -> tuple[float, float, int, float]:
    """Exact permutation test on |mean(group A) - mean(group B)| over
    per-model scalars (mirrors risk_drift_diff_attackers.group_label_permutation_test,
    which does the same thing but for pairwise distance matrices instead).
    """
    model_ids = sorted(values)
    group_names = sorted(set(group_of.values()))
    group_sizes = [sum(1 for m in model_ids if group_of[m] == g) for g in group_names]
    n = len(model_ids)
    total_assignments = _assignment_count(n, group_sizes)

    def stat(labels: list[int]) -> float:
        by_group: dict[int, list[float]] = {}
        for i, lab in enumerate(labels):
            by_group.setdefault(lab, []).append(values[model_ids[i]])
        means = [np.mean(v) for v in by_group.values()]
        return abs(means[0] - means[1])

    name_to_index = {name: i for i, name in enumerate(group_names)}
    observed_labels = [name_to_index[group_of[m]] for m in model_ids]
    observed = stat(observed_labels)

    extreme = sum(
        1 for labels in _multiset_assignments(n, group_sizes)
        if stat(labels) >= observed - 1e-12
    )
    p_value = extreme / total_assignments
    min_p = _label_symmetry_factor(group_sizes) / total_assignments
    return observed, p_value, total_assignments, min_p


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="rat_bench")
    parser.add_argument("--data-in", default="data/rat_bench")
    parser.add_argument("--tri-risk-dir", type=Path, default=DATA_DIR / "rat_bench" / "tri_risk")
    args = parser.parse_args()

    gold = load_gold_direct_spans(args.dataset, args.data_in)
    n_with_gold = sum(1 for spans in gold.values() if spans)
    print(f"Records with >=1 locatable direct identifier: {n_with_gold}/{len(gold)}")

    bart_paths = sorted(args.tri_risk_dir.glob("shap.jsonl")) + sorted(args.tri_risk_dir.glob("shap_model_*.jsonl"))
    nobart_paths = sorted(args.tri_risk_dir.glob("shap_nobart.jsonl")) + sorted(args.tri_risk_dir.glob("shap_nobart_model_*.jsonl"))

    group_of: dict[str, str] = {}
    mean_auc_by_model: dict[str, float] = {}
    for group_name, paths in (("bart", bart_paths), ("nobart", nobart_paths)):
        for path in paths:
            model_id = f"{group_name}:{path.stem}"
            shap = load_shap(path)
            aucs, skipped_no_gold, skipped_degenerate = per_record_gold_auc(shap, gold)
            group_of[model_id] = group_name
            mean_auc_by_model[model_id] = float(aucs.mean())
            print(
                f"{model_id}: n={len(aucs)} mean_AUC={aucs.mean():.4f} std={aucs.std():.4f} "
                f"(skipped no-gold={skipped_no_gold}, degenerate={skipped_degenerate})"
            )

    for group_name in ("bart", "nobart"):
        vals = [v for m, v in mean_auc_by_model.items() if group_of[m] == group_name]
        print(f"\n{group_name}: mean-of-model-means={np.mean(vals):.4f} std={np.std(vals):.4f} (n_models={len(vals)})")

    observed, p_value, total_assignments, min_p = group_mean_permutation_test(mean_auc_by_model, group_of)
    print("\nExact group-label permutation test on |mean(bart) - mean(nobart)| gold-span AUC:")
    print(f"  observed = {observed:.4f}")
    print(f"  p-value = {p_value:.4f} (min achievable = {min_p:.4f}, {total_assignments} label assignments enumerated)")


if __name__ == "__main__":
    main()
