import json
from pathlib import Path

import torch
from torch.utils.data import Dataset


def load_jsonl(path, skip_first_lines: int = 0):
    with open(path, 'r') as f:
        return [json.loads(line) for line in f]


class Data(Dataset):
    def __init__(self, data_path: Path):
        self.data = load_jsonl(data_path, skip_first_lines=1)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.tensor(self.data[idx], dtype=torch.float32)
