import os
import random
from typing import Any
from datasets import load_from_disk

import json

def get_permutation_of_length(n: int, k: int) -> list[int]:
    perm = list(range(n))
    random.shuffle(perm)
    return perm[:k]

def sample_data(data: list[dict[str, Any]], k: int) -> list[dict[str, Any]]:
    if k >= len(data):
        return data
    indices = get_permutation_of_length(len(data), k)
    return [data[i] for i in indices]

def get_tab_data() -> list[dict[str, Any]]:
    with open("data/TAB/tab.json", "r", encoding="utf-8") as f:
        return json.load(f)

def get_db_bio_data() -> list[dict[str, Any]]:
    db_bio_data_nested = [
        load_from_disk("data/db_bio/train"),
        load_from_disk("data/db_bio/validation"),
        load_from_disk("data/db_bio/test"),
    ]
    return [x for db_bio in db_bio_data_nested for x in db_bio]

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Sample data from datasets.")
    parser.add_argument("--data", "-d", type=str, choices=["tab", "db_bio"], required=True, help="Dataset to sample from")
    parser.add_argument("--n_samples", "-n", type=int, default=5, help="Number of samples to retrieve")
    parser.add_argument("--small_size", type=int, default=50, help="Smaller set of individuals")
    parser.add_argument("--large_size", type=int, default=500, help="Larger set of individuals")
    
    args = parser.parse_args()
    n_samples = args.n_samples
    if args.data == "tab":
        data = get_tab_data()
    elif args.data == "db_bio":
        data = get_db_bio_data()
    else:
        raise ValueError(f"Unsupported dataset: {args.data}")
    all_indices = list(range(len(data)))

    os.makedirs(f"indices/{args.data}/test", exist_ok=True)
    os.makedirs(f"indices/{args.data}/train", exist_ok=True)
    os.makedirs(f"indices/{args.data}/val", exist_ok=True)
    for i in range(n_samples):
        large_indices = sample_data(all_indices, args.large_size)
        small_indices = sample_data(large_indices, args.small_size)
        train_val_indices = list(set(all_indices) - set(large_indices))
        train_size = int(0.8 * len(train_val_indices))
        train_indices = sample_data(train_val_indices, train_size)
        val_indices = list(set(train_val_indices) - set(train_indices))

        with open(f"indices/{args.data}/test/{args.small_size}_eval_{i}.txt", "w") as f:
            f.write("\n".join(str(idx) for idx in small_indices) + "\n")
        with open(f"indices/{args.data}/test/{args.large_size}_random_{i}.txt", "w") as f:
            f.write("\n".join(str(idx) for idx in large_indices) + "\n")
        with open(f"indices/{args.data}/train/{i}.txt", "w") as f:
            f.write("\n".join(str(idx) for idx in train_indices) + "\n")
        with open(f"indices/{args.data}/val/{i}.txt", "w") as f:
            f.write("\n".join(str(idx) for idx in val_indices) + "\n")
