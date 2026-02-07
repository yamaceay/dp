import re
import json
import os
from pathlib import Path

from dp.runtime import load_runtime_bundle

texts_processed_expected = {
    "tab": 127,
    "reddit": 525,
}
pdirs = {
    "tab": [
        "tab_lrec1",
        "tab_lrec2_1",
        "tab_lrec2_2",
        "tab_lrec2_3",
        "tab_lrec2_4",
        "tab_lrec2_5",
        "tab_lrec2_6",
        "tab_lrec2_7",
    ],
    "reddit": [
        "reddit_lrec1",
        "reddit_lrec2_1",
        "reddit_lrec2_2",
        "reddit_lrec2_3",
        "reddit_lrec2_4",
        "reddit_lrec2_5",
        "reddit_lrec2_6",
    ],
}
files = [(dataset, pdir, "logs/" + pdir + "/" + f) for dataset, pdirs in pdirs.items() for pdir in pdirs for f in os.listdir("logs/" + pdir) if f.endswith(".out")]
cmd_pattern = r"\[[0-9]{4}-[0-9]{2}-[0-9]{2} [0-9]{2}:[0-9]{2}:[0-9]{2}\] Running command: scripts/task.sh --incr --state slurm/states/.*?/(.*?).state"
time_pattern = r"Total time: (\d+\.\d+)s\s+Texts processed: (\d+)\s+Average time per text: (\d+\.\d+)s\s+Throughput: (\d+\.\d+) texts/s"
unique_name_pattern = r"--unique_name (.*?)_(.+?)(?:(?:_k|_risk|_pii)?(?:_eps_[0-9]+)?)?$"
runtime_pattern = r"--runtime_in (.+?) --output"


def resolve_runtime_params(runtime_in: list[str]) -> dict[str, float | int]:
    if not runtime_in:
        return {}
    bundle = load_runtime_bundle(runtime_in)
    resolved: dict[str, float | int] = {}
    if bundle.epsilon_value is not None:
        resolved["epsilon"] = int(bundle.epsilon_value)
    if bundle.k_values:
        resolved["k"] = max(int(v) for v in bundle.k_values)
    if bundle.risk_tolerance_values:
        resolved["rho"] = min(float(v) for v in bundle.risk_tolerance_values)
    if bundle.pii_confidence_values:
        resolved["lambda"] = min(float(v) for v in bundle.pii_confidence_values)
    return resolved

time_stats = []
for dataset, pdir, file in files:
    with open(file, "r") as f:
        content = f.read()

        cmd_match = re.search(cmd_pattern, content, re.MULTILINE)
        if not cmd_match:
            raise ValueError(f"No command found in {file}")
        command = cmd_match.group(1)

        with open("slurm/states/" + pdir + "/" + command + ".state", "r") as sf:
            real_command = sf.readlines()[2].strip()

            get_unique_name_match = re.search(unique_name_pattern, real_command)
            if not get_unique_name_match:
                raise ValueError(f"No unique_name found in {file}")
            dataset_name = get_unique_name_match.group(1)
            if dataset_name != dataset:
                raise ValueError(f"Dataset name mismatch in {file}: {dataset_name} vs {dataset}")
            unique_name = get_unique_name_match.group(2)

            get_runtime_in_match = re.search(runtime_pattern, real_command)
            runtime_in = []
            if get_runtime_in_match:
                runtime_in = get_runtime_in_match.group(1).split(" ")
            runtimes_grouped = resolve_runtime_params(runtime_in)

        matches = re.search(time_pattern, content, re.MULTILINE)
        if not matches:
            raise ValueError(f"No time stats found in {file}")
        
        total_time = float(matches.group(1))
        texts_processed = int(matches.group(2))
        avg_time_per_text = float(matches.group(3))
        throughput = float(matches.group(4))

        if texts_processed != texts_processed_expected[dataset]:
            raise ValueError(f"Texts processed mismatch in {file}: {texts_processed} vs {texts_processed_expected[dataset]}")

        time_stats.append({
            "dataset": dataset_name,
            "method": unique_name,
            "params": runtimes_grouped,
            "runtime_total_seconds": total_time,
            "runtime_mean_seconds_per_text": avg_time_per_text,
            "runtime_mean_texts_per_second": throughput,
        })

time_stats_by_dataset = {}
for stats in time_stats:
    dataset = stats["dataset"]
    time_stats_by_dataset.setdefault(dataset, []).append(stats)

for dataset, stats in time_stats_by_dataset.items():
    with open(f"visualize/pretty/{dataset}/runtime.json", "w") as out_f:
        json.dump(stats, out_f, indent=2)
