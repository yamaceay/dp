import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List

from dp.utils.log_keys import parse_method_and_params_from_source_path

REQUIRES_FULL = [
    {"method": "petre_shap", "params": ["k"]},
    {"method": "dpmlm_shap", "params": ["epsilon", "k"]},
]
REQUIRES_FULL_DEID = [
    {"method": "petre_shap", "params": ["k"]},
    {"method": "risk", "params": ["rho"]},
    {"method": "dpmlm_shap", "params": ["epsilon"]},
    {"method": "dpmlm_shap", "params": ["epsilon", "k"]},
    {"method": "dpmlm_shap", "params": ["epsilon", "rho"]},
    {"method": "dpmlm_shap", "params": ["epsilon", "lambda"]},
    {"method": "dpmlm_shap_no_presidio", "params": ["epsilon"]},
]

STEP_1_PERF_PATTERN = re.compile(
    r"Anonymization Performance:\s*\n"
    r"\s*Total time:\s*([\d.]+)s\s*\n"
    r"\s*Texts processed:\s*(\d+)\s*\n"
    r"\s*Average time per text:\s*([\d.]+)s",
)
STEP_1_OUTPUT_BLOCK_PATTERN = re.compile(r"Output written to:\s*(.*?\.jsonl)", re.DOTALL)
STEP_1_DATASET_PATTERN = re.compile(r"_(db_bio|tab|rat_bench)_")
STEP_1_LOG_DIRS: List[str] = [
    "logs/0_db_bio_simple_anonymization",
    "logs/0_tab_simple_anonymization",
    "logs/0_rat_bench_simple_anonymization",
    "logs/4_db_bio_further_anonymization",
    "logs/4_tab_further_anonymization",
    "logs/4_rat_bench_further_anonymization"
]

STEP_2_PERF_PATTERN = re.compile(
    r"(?:^|[\r\n])(?:[^\r\n]*:\s*)?"
    r"100%\|[^|\r\n]*\|\s*(\d+)/(\d+)\s+\[((?:\d+:)?\d{1,2}:\d{2})<00:00,\s*[\d.]+s/it\]"
)
STEP_2_NAME_PATTERN = re.compile(
    r"(?:Executing \(task_id=\d+\):|Base command:)\s*python data_for_tri\.py --config "
    r"configs/1_tri_preprocessing/(db_bio|tab|rat_bench)/(full|full_deid)\.yaml"
    r"(?:\s+--task_id(?:\s+\d+|\s+\{task_id\}))?"
)
STEP_2_LOG_DIRS: List[str] = [
    "logs/1_db_bio_tri_preprocessing",
    "logs/1_tab_tri_preprocessing",
    "logs/1_rat_bench_tri_preprocessing",
]

STEP_3_TRAIN_RUNTIME_PATTERN = re.compile(r"'train_runtime':\s*([0-9]+(?:\.[0-9]+)?)")
STEP_3_NAME_PATTERN = re.compile(
    r"(?:Executing \(task_id=\d+\):|Base command:)\s*python tri_by_bk\.py --mode train --training-in "
    r"configs/2_tri_training/(db_bio|tab|rat_bench)/(full|full_deid)\.yaml "
    r"--run-name (?:(full|full_deid)) --set training.lr=5e-5 --set training.debug_tri=true"
)
STEP_3_LOG_DIRS: List[str] = [
    "logs/2_db_bio_tri_training",
    "logs/2_tab_tri_training",
    "logs/2_rat_bench_tri_training",
]

STEP_4_RISK_RUNTIME_PATTERN = re.compile(
    r"(?:^|[\r\n])(?:[^\r\n]*:\s*)?"
    r"100%\|[^|\r\n]*\|\s*(\d+)/(\d+)\s+\[((?:\d+:)?\d{1,2}:\d{2})<00:00,\s*[\d.]+s/it\]"
)
STEP_4_NAME_PATTERN = re.compile(
    r"(?:Executing \(task_id=\d+\):|Base command:)\s*python risk\.py --config "
    r"configs/3_tri_risk_precomputation/(db_bio|tab|rat_bench)/shap\.yaml"
)
STEP_4_LOG_DIRS: List[str] = [
    "logs/3_db_bio_risk_precomputation",
    "logs/3_tab_risk_precomputation",
    "logs/3_rat_bench_risk_precomputation",
]

def read_jsonl_rows(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                print(f"Skipping invalid JSON line in {path}: {line}", file=sys.stderr)
    return rows


def write_jsonl_rows(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            json.dump(row, f, ensure_ascii=False)
            f.write("\n")
    print(f"Wrote {len(rows)} entries to {path}")


def parse_duration_to_seconds(total_time: str) -> int:
    parts = [int(part) for part in total_time.split(":")]
    if len(parts) == 2:
        minutes, seconds = parts
        return minutes * 60 + seconds
    if len(parts) == 3:
        hours, minutes, seconds = parts
        return hours * 3600 + minutes * 60 + seconds
    raise ValueError(f"Unexpected duration format: {total_time}")


def matches_method_family(method: str, required_method: str) -> bool:
    return method == required_method or method.startswith(f"{required_method}_")


def matches_requirement(method: str, params: Dict[str, Any], requirement: Dict[str, Any]) -> bool:
    return matches_method_family(method, requirement["method"]) and set(params) == set(requirement["params"])


def matches_exact_requirement(method: str, params: Dict[str, Any], requirement: Dict[str, Any]) -> bool:
    return method == requirement["method"] and set(params) == set(requirement["params"])


def run_step_0(root: Path) -> None:
    for dataset in ["db_bio", "tab", "rat_bench"]:
        anon_path = root / "logs" / "runtime" / dataset / "anon.jsonl"
        if not anon_path.is_file():
            print(f"skipping missing anon log: {anon_path}", file=sys.stderr)
            continue
        rows = read_jsonl_rows(anon_path)
        presidio_runtime = next(row for row in rows if row["method"] == "presidio").get("anon_total_time_s")
        if presidio_runtime is None:
            raise ValueError(f"Presidio runtime not found in {anon_path}")
        new_rows = []
        for row in rows:
            method = row["method"]
            params = row["params"]
            new_row = dict(
                method=method, 
                params=params, 
                deid_total_time_s=0.0,
            )
            if any(matches_requirement(method, params, req) for req in REQUIRES_FULL_DEID):
                new_row["deid_total_time_s"] = presidio_runtime
            new_rows.append(new_row)
            
        write_jsonl_rows(root / "logs" / "runtime" / dataset / "deid.jsonl", new_rows)

def run_step_1(root: Path) -> None:
    def extract_output_paths(text: str) -> List[str]:
        paths: List[str] = []
        for match in STEP_1_OUTPUT_BLOCK_PATTERN.finditer(text):
            path = "".join(match.group(1).split())
            if path.endswith(".jsonl"):
                paths.append(path)
        return paths

    def parse_dataset_from_output_path(output_path: str) -> str | None:
        match = STEP_1_DATASET_PATTERN.search(output_path)
        if match is None:
            return None
        return match.group(1)

    def extract_from_file(path: Path) -> List[Dict[str, Any]]:
        text = path.read_text(errors="replace")
        perf = STEP_1_PERF_PATTERN.search(text)
        if not perf:
            return []
        total_time = float(perf.group(1))
        texts_processed = int(perf.group(2))
        rows: List[Dict[str, Any]] = []
        for output_path in extract_output_paths(text):
            dataset = parse_dataset_from_output_path(output_path)
            if dataset is None:
                continue
            method, params = parse_method_and_params_from_source_path(output_path, dataset)
            rows.append({
                "method": method,
                "params": params,
                "anon_total_time_s": total_time,
                "anon_texts_processed": texts_processed,
            })
        return rows

    dataset_rows: Dict[str, List[Dict[str, Any]]] = {}
    for log_dir in STEP_1_LOG_DIRS:
        directory = root / log_dir
        if not directory.is_dir():
            print(f"skipping missing dir: {directory}", file=sys.stderr)
            continue
        for path in sorted(directory.glob("*.out")):
            for row in extract_from_file(path):
                if "db_bio" in str(directory):
                    dataset = "db_bio"
                elif "tab" in str(directory):
                    dataset = "tab"
                elif "rat_bench" in str(directory):
                    dataset = "rat_bench"
                else:
                    continue
                dataset_rows.setdefault(dataset, []).append(row)

    for dataset, rows in dataset_rows.items():
        write_jsonl_rows(root / "logs" / "runtime" / dataset / "anon.jsonl", rows)


def run_step_2(root: Path) -> None:
    def extract_from_file(path: Path, directory: str) -> Dict[str, Any] | None:
        text = path.read_text(errors="replace")
        perf = STEP_2_PERF_PATTERN.search(text)
        if not perf:
            return None
        texts_processed = int(perf.group(1))
        total_items = int(perf.group(2))
        if texts_processed == 0 or texts_processed != total_items:
            return None
        name = STEP_2_NAME_PATTERN.search(text)
        if not name:
            return None
        dataset = name.group(1)
        if dataset not in directory:
            return None
        return {
            "dataset": dataset,
            "type_of_bk": name.group(2),
            "bk_gen_total_time_s": parse_duration_to_seconds(perf.group(3)),
        }

    runtime_by_dataset: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for log_dir in STEP_2_LOG_DIRS:
        directory = root / log_dir
        if not directory.is_dir():
            print(f"skipping missing dir: {directory}", file=sys.stderr)
            continue
        for path in sorted(directory.glob("*.err")):
            runtime = extract_from_file(path, str(directory))
            if runtime is None:
                continue
            runtime_by_dataset.setdefault(runtime["dataset"], {})[runtime["type_of_bk"]] = runtime

    for dataset, dataset_runtime in runtime_by_dataset.items():
        anon_path = root / "logs" / "runtime" / dataset / "anon.jsonl"
        if not anon_path.is_file():
            print(f"skipping missing anon log: {anon_path}", file=sys.stderr)
            continue
        full_runtime = dataset_runtime.get("full")
        full_deid_runtime = dataset_runtime.get("full_deid")
        rows: List[Dict[str, Any]] = []
        for anon_row in read_jsonl_rows(anon_path):
            method = anon_row["method"]
            params = anon_row["params"]
            bk_gen_total_time = 0.0
            if full_runtime is not None and any(matches_requirement(method, params, req) for req in REQUIRES_FULL):
                bk_gen_total_time += full_runtime["bk_gen_total_time_s"]
            if full_deid_runtime is not None and any(matches_requirement(method, params, req) for req in REQUIRES_FULL_DEID):
                bk_gen_total_time += full_deid_runtime["bk_gen_total_time_s"]
            rows.append({
                "method": method,
                "params": params,
                "bk_gen_total_time_s": bk_gen_total_time,
            })
        write_jsonl_rows(root / "logs" / "runtime" / dataset / "bk_gen.jsonl", rows)


def run_step_3(root: Path) -> None:
    def extract_from_file(path: Path, directory: str) -> Dict[str, Any] | None:
        text = path.read_text(errors="replace")
        train_runtime_match = STEP_3_TRAIN_RUNTIME_PATTERN.search(text)
        if not train_runtime_match:
            return None
        name = STEP_3_NAME_PATTERN.search(text)
        if not name:
            return None
        dataset = name.group(1)
        if dataset not in directory:
            return None
        return {
            "dataset": dataset,
            "type_of_bk": name.group(2),
            "train_total_time_s": float(train_runtime_match.group(1)),
        }

    runtime_by_dataset: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for log_dir in STEP_3_LOG_DIRS:
        directory = root / log_dir
        if not directory.is_dir():
            print(f"skipping missing dir: {directory}", file=sys.stderr)
            continue
        for path in sorted(directory.glob("*.out")):
            runtime = extract_from_file(path, str(directory))
            if runtime is None:
                continue
            runtime_by_dataset.setdefault(runtime["dataset"], {})[runtime["type_of_bk"]] = runtime

    for dataset, dataset_runtime in runtime_by_dataset.items():
        anon_path = root / "logs" / "runtime" / dataset / "anon.jsonl"
        if not anon_path.is_file():
            print(f"skipping missing anon log: {anon_path}", file=sys.stderr)
            continue
        full_runtime = dataset_runtime.get("full")
        full_deid_runtime = dataset_runtime.get("full_deid")
        rows: List[Dict[str, Any]] = []
        for anon_row in read_jsonl_rows(anon_path):
            method = anon_row["method"]
            params = anon_row["params"]
            train_total_time = 0.0
            if full_runtime is not None and any(matches_requirement(method, params, req) for req in REQUIRES_FULL):
                train_total_time += full_runtime["train_total_time_s"]
            if full_deid_runtime is not None and any(matches_requirement(method, params, req) for req in REQUIRES_FULL_DEID):
                train_total_time += full_deid_runtime["train_total_time_s"]
            rows.append({
                "method": method,
                "params": params,
                "train_total_time_s": train_total_time,
            })
        write_jsonl_rows(root / "logs" / "runtime" / dataset / "train.jsonl", rows)


def run_step_4(root: Path) -> None:
    def extract_from_file(path: Path, directory: str) -> Dict[str, Any] | None:
        text = path.read_text(errors="replace")
        risk_runtime_match = STEP_4_RISK_RUNTIME_PATTERN.search(text)
        if not risk_runtime_match:
            return None
        texts_processed = int(risk_runtime_match.group(1))
        total_items = int(risk_runtime_match.group(2))
        if texts_processed == 0 or texts_processed != total_items:
            return None
        name = STEP_4_NAME_PATTERN.search(text)
        if not name:
            return None
        dataset = name.group(1)
        if dataset not in directory:
            return None
        return {
            "dataset": dataset,
            "risk_total_time_s": parse_duration_to_seconds(risk_runtime_match.group(3)),
        }

    runtime_by_dataset: Dict[str, Dict[str, Any]] = {}
    for log_dir in STEP_4_LOG_DIRS:
        directory = root / log_dir
        if not directory.is_dir():
            print(f"skipping missing dir: {directory}", file=sys.stderr)
            continue
        for path in sorted(directory.glob("*.err")):
            runtime = extract_from_file(path, str(directory))
            if runtime is None:
                continue
            runtime_by_dataset[runtime["dataset"]] = runtime

    for dataset, dataset_runtime in runtime_by_dataset.items():
        anon_path = root / "logs" / "runtime" / dataset / "anon.jsonl"
        if not anon_path.is_file():
            print(f"skipping missing anon log: {anon_path}", file=sys.stderr)
            continue
        rows: List[Dict[str, Any]] = []
        for anon_row in read_jsonl_rows(anon_path):
            method = anon_row["method"]
            params = anon_row["params"]
            risk_total_time = 0.0
            if any(matches_requirement(method, params, req) for req in REQUIRES_FULL_DEID):
                risk_total_time = dataset_runtime["risk_total_time_s"]
            rows.append({
                "method": method,
                "params": params,
                "risk_total_time_s": risk_total_time,
            })
        write_jsonl_rows(root / "logs" / "runtime" / dataset / "risk.jsonl", rows)


def main() -> None:
    root = Path(__file__).resolve().parent
    run_step_1(root)
    run_step_2(root)
    run_step_3(root)
    run_step_4(root)
    run_step_0(root)


if __name__ == "__main__":
    main()
