# src/data_prep/split_dataset.py
#
# Combine tasks and split into train/dev/test.

import json
import random


def split_dataset(input_files, output_dir, train_ratio=0.7, dev_ratio=0.15):
    tasks = []
    for file in input_files:
        with open(file, "r", encoding="utf-8") as f:
            tasks.extend(json.load(f))

    random.shuffle(tasks)
    n = len(tasks)
    train_end = int(n * train_ratio)
    dev_end = train_end + int(n * dev_ratio)

    splits = {
        "train": tasks[:train_end],
        "dev": tasks[train_end:dev_end],
        "heldout": tasks[dev_end:],
    }

    for split, data in splits.items():
        with open(f"{output_dir}/{split}.json", "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)


if __name__ == "__main__":
    input_files = [
        "data/tasks/trace_tasks.json",
        "data/tasks/synthetic_pairs.json",
        "data/tasks/adversarial_cases.json",
    ]
    split_dataset(input_files, "data/splits")
