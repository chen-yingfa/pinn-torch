from typing import List, Tuple
from pathlib import Path
import random
import json

import torch
from torch.utils.data import Dataset


class PinnDataset(Dataset):
    def __init__(self, data: List[List[float]]):
        self.data = data
        self.examples = torch.tensor(
            data, dtype=torch.float32, requires_grad=True
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        headers = ["t", "x", "y", "p", "u", "v"]
        return {key: self.examples[idx, i] for i, key in enumerate(headers)}


def load_jsonl(path, skip_first_lines: int = 0):
    with open(path, "r") as f:
        for _ in range(skip_first_lines):
            next(f)
        return [json.loads(line) for line in f]


def dump_json(path, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=4)


def get_dataset(data_path: Path) -> Tuple[PinnDataset, PinnDataset]:
    data = load_jsonl(data_path, skip_first_lines=1)
    random.shuffle(data)

    # It's weird that the test data is a subset of train data, but
    # that's what the original paper does.
    split_idx = int(len(data) * 0.9)
    train_data = data
    test_data = data[split_idx:]

    # train_data = train_data[:10000]
    # train_data = train_data[:1000]

    train_data = PinnDataset(train_data)
    test_data = PinnDataset(test_data)
    return train_data, test_data
