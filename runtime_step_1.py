import re
import csv
import sys
from pathlib import Path
from typing import List, Dict


PERF_PATTERN = re.compile(
    r"Anonymization Performance:\s*\n"
    r"\s*Total time:\s*([\d.]+)s\s*\n"
    r"\s*Texts processed:\s*(\d+)\s*\n"
    r"\s*Average time per text:\s*([\d.]+)s",
)

OUTPUT_PATTERN = re.compile(
    r"Output written to:\s*\S+/"
    r"\d{8}_\d{6}"
    r"_(db_bio|tab)"
    r"_([\w.]+?)"
    r"(?:\?([^\s.]+))?"
    r"\.jsonl"
)

LOG_DIRS: List[str] = [
    "logs/0_db_bio_simple_anonymization",
    "logs/0_tab_simple_anonymization",
    "logs/4_db_bio_further_anonymization",
    "logs/4_tab_further_anonymization",
]


def parse_params(raw: str | None) -> Dict[str, str]:
    if not raw:
        return {}
    return dict(pair.split("=", 1) for pair in raw.split("&") if "=" in pair)


def extract_from_file(path: Path) -> List[Dict[str, str]]:
    text = path.read_text(errors="replace")
    perf = PERF_PATTERN.search(text)
    if not perf:
        return []
    total_time = perf.group(1)
    texts_processed = perf.group(2)
    avg_time = perf.group(3)

    outputs = OUTPUT_PATTERN.findall(text)
    if not outputs:
        return []

    rows: List[Dict[str, str]] = []
    for dataset, method, params_raw in outputs:
        params = parse_params(params_raw)
        rows.append({
            "log_file": path.name,
            "dataset": dataset,
            "method": method,
            "total_time_s": total_time,
            "texts_processed": texts_processed,
            "avg_time_s": avg_time,
            **params,
        })
    return rows


def main() -> None:
    root = Path(__file__).parent
    all_rows: List[Dict[str, str]] = []
    all_keys: set[str] = set()

    for log_dir in LOG_DIRS:
        d = root / log_dir
        if not d.is_dir():
            print(f"skipping missing dir: {d}", file=sys.stderr)
            continue
        for f in sorted(d.glob("*.out")):
            rows = extract_from_file(f)
            for row in rows:
                all_keys.update(row.keys())
            all_rows.extend(rows)

    fixed_cols = ["log_file", "dataset", "method", "total_time_s", "texts_processed", "avg_time_s"]
    param_cols = sorted(all_keys - set(fixed_cols))
    columns = fixed_cols + param_cols

    out_path = root / "runtime_step_1_results.csv"
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"extracted {len(all_rows)} entries -> {out_path}")


if __name__ == "__main__":
    main()