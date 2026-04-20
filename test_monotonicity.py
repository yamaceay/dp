from __future__ import annotations

import argparse
import json
from pathlib import Path

MANIFEST_DIR = Path("images/mega_metrics")

PARAM_ORDER: dict[str, list] = {
    "epsilon": [10, 25, 50, 100, 250],
    "k":       [2, 3, 5, 7, 10],
    "rho":     [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    "lambda":  [0.8, 0.9, 0.95, 0.98, 0.99],
}

# Thesis claims — how each metric should respond as the parameter increases:
#   k ↑        more tokens masked    → P ↓  TRIR ↓  U ↓  SS ↓  D ↑
#   rho ↑      stricter risk stop    → P ↑  TRIR ↑  U ↑  SS ↑  D ↓  (less masking)
#   lambda ↑   stricter DP stop      → P ↑  TRIR ↑  U ↑  SS ↑  D ↓  (less masking)
EXPECTED_DIRECTION: dict[str, dict[str, int]] = {
    "P":    {"k": -1, "rho": +1, "lambda": +1},
    "TRIR": {"k": -1, "rho": +1, "lambda": +1},
    "U":    {"k": -1, "rho": +1, "lambda": +1},
    "SS":   {"k": -1, "rho": +1, "lambda": +1},
    "D":    {"k": +1, "rho": -1, "lambda": -1},
}


def _rank(param: str, value: object) -> int:
    try:
        return PARAM_ORDER[param].index(value)
    except (KeyError, ValueError):
        return -1


def _is_monotone(values: list[float], direction: int) -> bool:
    for a, b in zip(values, values[1:]):
        if direction == +1 and b < a:
            return False
        if direction == -1 and b > a:
            return False
    return True


def _load_combinations(metric: str, label_prefix: str | None) -> dict[str, list[dict]]:
    path = MANIFEST_DIR / f"{metric.lower()}_manifest.json"
    if not path.exists():
        return {}
    with open(path) as f:
        manifest = json.load(f)
    result: dict[str, list[dict]] = {}
    for dataset, variants in manifest.items():
        combos: list[dict] = []
        for variant in variants:
            if label_prefix and not variant["label"].startswith(label_prefix):
                continue
            combos.extend(variant["combinations"])
        if combos:
            result[dataset] = combos
    return result


def check_metric(metric: str, label_prefix: str | None) -> tuple[int, int]:
    data = _load_combinations(metric, label_prefix)
    if not data:
        print(f"  [SKIP] {metric}: no data")
        return 0, 0

    directions = EXPECTED_DIRECTION[metric]
    epsilons = PARAM_ORDER["epsilon"]
    sweep_params = [p for p in ("k", "rho", "lambda") if p in directions]

    total_pass, total_fail = 0, 0
    metric_failures: list[str] = []

    for eps in epsilons:
        param_results: list[str] = []
        eps_pass = True

        for param in sweep_params:
            direction = directions[param]
            param_pass = True

            for dataset, combos in data.items():
                pairs = [
                    (c[param], float(c[metric]))
                    for c in combos
                    if c.get("epsilon") == eps and param in c and c.get(metric) is not None
                ]
                if len(pairs) < 2:
                    continue

                ordered = sorted(pairs, key=lambda t: _rank(param, t[0]))
                vals = [v for _, v in ordered]

                if _is_monotone(vals, direction):
                    total_pass += 1
                else:
                    total_fail += 1
                    param_pass = False
                    metric_failures.append(
                        f"      ε={eps:<4}  {param} ↑  {dataset}"
                        f"  values={[round(v, 4) for v in vals]}"
                    )

            if param_pass:
                param_results.append(f"{param}:✓")
            else:
                eps_pass = False
                param_results.append(f"{param}:✗")

        status = "✓" if eps_pass else "✗"
        print(f"    {status}  ε={eps:<4}  {'  '.join(param_results)}")

    total = total_pass + total_fail
    arrow = "✓" if total_fail == 0 else "✗"
    print(f"  {arrow}  {metric:5s}  {total_pass}/{total} monotone")
    for line in metric_failures:
        print(line)

    return total_pass, total_fail


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", metavar="PREFIX", help="filter variants by label prefix")
    args = parser.parse_args()

    header = "Monotonicity check — thesis claims"
    if args.label:
        header += f"  (label: {args.label})"
    print(header + "\n")

    total_pass, total_fail = 0, 0
    for metric in ["P", "TRIR", "U", "SS", "D"]:
        p, f = check_metric(metric, label_prefix=args.label)
        total_pass += p
        total_fail += f
        print()

    if total_fail == 0:
        print(f"PASS  all {total_pass} checks are monotone as claimed in the thesis.")
    else:
        print(f"FAIL  {total_fail} of {total_pass + total_fail} checks violated monotonicity.")


if __name__ == "__main__":
    main()
