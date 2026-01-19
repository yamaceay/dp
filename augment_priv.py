import json

with open("logs/tab_priv_exp.jsonl", "r") as f_in, open("logs/tab_priv_exp_patched.jsonl", "w") as f_out:
    all_lines = f_in.readlines()
    ranks_per_method = {}
    ranks_original = []
    for line in all_lines:
        record = json.loads(line)
        if record["type"] == "original_rank":
            ranks_original.append(record["rank"])
        elif record["type"] == "evaluation_rank":
            ranks_per_method.setdefault(record["evaluation"], []).append(record["rank"])

    accuracies_per_method = {k: sum(v_i == 1 for v_i in v) / len(v) for k, v in ranks_per_method.items()}
    accuracy_original = sum(v_i == 1 for v_i in ranks_original) / len(ranks_original)
    for line in all_lines:
        record = json.loads(line)
        if record["type"] == "experiment":
            record.setdefault("score", {}).update({"accuracy": accuracy_original})
        elif record["type"] == "evaluation":
            accuracy = accuracies_per_method[record["name"]]
            record.setdefault("summary", {}).update({"accuracy": accuracy})
        f_out.write(json.dumps(record) + "\n")