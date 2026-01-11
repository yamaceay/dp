import numpy as np
import json

with open("logs/reddit_priv_exp.jsonl", "r") as f:
    all_records = [json.loads(line) for line in f.readlines()]
    evaluation_dict = {}
    original_dict = []
    for i, record in enumerate(all_records):
        if record["type"] == "evaluation_rank":
            evaluation_dict.setdefault(record["evaluation"], []).append(record)
        elif record["type"] == "original_rank":
            original_dict.append(record)

    mrr_original = sum([1 / x["rank"] for x in original_dict]) / len(original_dict)

    mrr_others = {}
    for eval_name, records in evaluation_dict.items():
        mean_reciprocal_rank = sum([1 / x["rank"] for x in records]) / len(records)
        mrr_others[eval_name] = mean_reciprocal_rank

    patched_records = []
    for i, record in enumerate(all_records):
        if record["type"] == "experiment":
            record["score"] = {"mean_reciprocal_rank": mrr_original}
        elif record["type"] == "evaluation":
            mean_reciprocal_rank = mrr_others[record["name"]]
            record["summary"]["mean_reciprocal_rank"] = mean_reciprocal_rank
        patched_records.append(record)

# with open("logs/reddit_priv_exp_patched.jsonl", "w") as f:
#     for record in patched_records:
#         f.write(json.dumps(record) + "\n")