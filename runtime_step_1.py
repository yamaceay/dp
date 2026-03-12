
import re
import json
import sys
from pathlib import Path
from typing import List, Dict, Any

from dp.utils.log_keys import parse_method_and_params_from_source_path

PERF_PATTERN = re.compile(
    r"Anonymization Performance:\s*\n"
    r"\s*Total time:\s*([\d.]+)s\s*\n"
    r"\s*Texts processed:\s*(\d+)\s*\n"
    r"\s*Average time per text:\s*([\d.]+)s",
)

OUTPUT_BLOCK_PATTERN = re.compile(r"Output written to:\s*(.*?\.jsonl)", re.DOTALL)
DATASET_PATTERN = re.compile(r"_(db_bio|tab)_")

LOG_DIRS: List[str] = [
    "logs/0_db_bio_simple_anonymization",
    "logs/0_tab_simple_anonymization",
    "logs/4_db_bio_further_anonymization",
    "logs/4_tab_further_anonymization",
]


def extract_output_paths(text: str) -> List[str]:
    paths: List[str] = []
    for match in OUTPUT_BLOCK_PATTERN.finditer(text):
        path = "".join(match.group(1).split())
        if path.endswith(".jsonl"):
            paths.append(path)
    return paths


def parse_dataset_from_output_path(output_path: str) -> str | None:
    match = DATASET_PATTERN.search(output_path)
    if match is None:
        return None
    return match.group(1)


def extract_from_file(path: Path) -> List[Dict[str, Any]]:
    text = path.read_text(errors="replace")
    perf = PERF_PATTERN.search(text)
    if not perf:
        return []
    total_time = float(perf.group(1))
    texts_processed = int(perf.group(2))
    avg_time = float(perf.group(3))

    output_paths = extract_output_paths(text)
    if not output_paths:
        return []

    rows: List[Dict[str, Any]] = []
    for output_path in output_paths:
        dataset = parse_dataset_from_output_path(output_path)
        if dataset is None:
            continue
        method, params = parse_method_and_params_from_source_path(output_path, dataset)
        rows.append({
            "method": method,
            "params": params,
            "anon_total_time_s": total_time,
            "anon_texts_processed": texts_processed,
            "anon_avg_time_s": avg_time,
        })
    return rows


def main() -> None:
    root = Path(__file__).parent
    dataset_rows: dict[str, list[dict[str, Any]]] = {}

    for log_dir in LOG_DIRS:
        d = root / log_dir
        if not d.is_dir():
            print(f"skipping missing dir: {d}", file=sys.stderr)
            continue
        for f in sorted(d.glob("*.out")):
            rows = extract_from_file(f)
            for row in rows:
                if "db_bio" in str(d):
                    dataset = "db_bio"
                elif "tab" in str(d):
                    dataset = "tab"
                else:
                    continue
                dataset_rows.setdefault(dataset, []).append(row)

    for dataset, rows in dataset_rows.items():
        out_dir = root / "logs" / "runtime" / dataset
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "anon.jsonl"
        with out_path.open("w", encoding="utf-8") as f:
            for row in rows:
                json.dump(row, f, ensure_ascii=False)
                f.write("\n")
        print(f"Wrote {len(rows)} entries to {out_path}")


if __name__ == "__main__":
    main()