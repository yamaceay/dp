from pathlib import Path
import re

base_path = 'logs/a2x_reddit_debug_tri'
suffix = '.out'
all_paths = list(Path(base_path).glob('*' + suffix))

pattern = r"Mean Reciprocal Rank \(MRR\): (.*?)\nAccuracy \(ACC\): (.*?)\n"

is_orig_or_deid = lambda i: ((i // 2) % 2, i % 2)

grouped_by_orig_deid = {}

for p in all_paths:
    with open(p, 'r') as f:
        content = f.read()

    p_small = str(p)[len(base_path + "/"):-len(suffix)]
    [_, i] = p_small.split("_", 1)
    model, data = is_orig_or_deid(int(i))
    model = "Original" if model == 0 else "De-identified"
    data = "Original" if data == 0 else "De-identified"

    matches = re.search(pattern, content, re.DOTALL)
    if matches:
        mrr = matches.group(1)
        acc = matches.group(2)
        # print(f"MRR: {mrr}, ACC: {acc}, Model: {model}, Data: {data}")
        key = (model, data)
        grouped_by_orig_deid.setdefault(key, []).append((mrr, acc, int(i)))
        # print("-----")

for key, values in grouped_by_orig_deid.items():
    values_sorted = sorted(values, key=lambda x: (x[0], x[1]), reverse=True)
    print(f"Model: {key[0]}, Data: {key[1]}")
    for mrr, acc, i in values_sorted:
        print(f"  MRR: {mrr}, ACC: {acc}, Index: {i}")
    print("=====")