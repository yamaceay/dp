import json
import re
import pandas as pd
import os

FILES = [
    ("logs/reddit_priv_exp.jsonl", "reddit"),
    # ("logs/tab_priv_exp.jsonl", "tab"),
]

def params_to_str_for_sort(params):
    str_parts = []
    for k, v in params.items():
        if k == "epsilon":
            str_parts.append((k, f"{1000 - v:03d}"))
        elif k == "k":
            str_parts.append((k, f"{v:02d}"))
        elif k in {"rho", "lambda"}:
            str_parts.append((k, f"{int(100 - v * 100):03d}"))
    return "&".join(f"{k}={v}" for k, v in str_parts)

def params_to_str(params):
    return "&".join(f"{k}={v}" for k, v in params.items())

def read_data(files):
    for file_path, dataset_name in files:
        with open(file_path) as file:
            for line in file:
                data = json.loads(line)
                if data["type"] != "evaluation":
                    continue

                key = re.sub(r"outputs/[a-z]+/[a-z]+/[0-9]{8}_[0-9]{6}_[a-z]+_(.*?).jsonl", r"\1", data["source"])
                key = re.sub(r"_eps_[0-9]{3}(\?.*?)", r"\1", key)
                key = re.sub(r"(?:_k|_risk|_pii)(\?.*?)", r"\1", key)

                params = {}
                if "?" in key:
                    key, params_str = key.split("?", 1)
                    for param in params_str.split("&"):
                        if "=" in param:
                            name, value = param.split("=", 1)
                            if name == "epsilon" or name == "k":
                                value = int(value)
                            elif name == "rho" or name == "lambda":
                                value = float(value) / 100.0
                            params[f"{name}"] = value
                        else: 
                            raise ValueError(f"Unexpected hyperparameter format: {param}")
                params = dict(sorted(params.items(), key=lambda item: item[0]))
                res = data["summary"]
                values = {
                    "privacy_mean_rank_change": res["mean"],
                    "privacy_num_rank_increased": res["improved"],
                    "privacy_num_rank_decreased": res["degraded"],
                }
                yield {"method": key, "params": params, "dataset": dataset_name, **values}

if __name__ == "__main__":
    entries = list(read_data(FILES))
    entries = sorted(entries, key=lambda x: (x["method"], params_to_str_for_sort(x["params"])))
    entries_by_dataset_by_method = {}
    for entry in entries:
        entry_copy = entry.copy()
        dataset = entry_copy.pop("dataset")
        method = entry_copy.pop("method")
        # entry_copy["params"] = params_to_str(entry_copy.pop("params"))
        if dataset not in entries_by_dataset_by_method:
            entries_by_dataset_by_method[dataset] = {}
        if method not in entries_by_dataset_by_method[dataset]:
            entries_by_dataset_by_method[dataset][method] = []
        entries_by_dataset_by_method[dataset][method].append(entry_copy)

    entries_by_dataset_by_params = {}
    for entry in entries:
        entry_copy = entry.copy()
        dataset = entry_copy.pop("dataset")
        params = params_to_str(entry_copy.pop("params"))
        if dataset not in entries_by_dataset_by_params:
            entries_by_dataset_by_params[dataset] = {}
        if params not in entries_by_dataset_by_params[dataset]:
            entries_by_dataset_by_params[dataset][params] = []
        entries_by_dataset_by_params[dataset][params].append(entry_copy)

    with open("visualize/reddit_summary.json", "w", encoding="utf-8") as f:
        json.dump(entries, f, indent=2)
    
    with open("visualize/reddit_summary_by_method.json", "w", encoding="utf-8") as f:
        json.dump(entries_by_dataset_by_method, f, indent=2)

    with open("visualize/reddit_summary_by_params.json", "w", encoding="utf-8") as f:
        json.dump(entries_by_dataset_by_params, f, indent=2)