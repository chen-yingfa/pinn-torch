import json
from pathlib import Path
from typing import List, Tuple
import random
from time import time

import torch
from torch.utils.data import Dataset, DataLoader

from model import Pinn


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


def get_dataset(data_path: Path) -> Tuple[PinnDataset, PinnDataset]:
    data = load_jsonl(data_path, skip_first_lines=1)
    random.shuffle(data)
    split_idx = int(len(data) * 0.9)
    train_data = data[:split_idx]
    test_data = data[split_idx:]
    train_data = PinnDataset(train_data)
    test_data = PinnDataset(test_data)
    return train_data, test_data


def main():
    torch.random.manual_seed(0)
    random.seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Hyperparameters
    hidden_dims = [20] * 8
    lr = 1e-3
    lr_step = 1  # Unit is epoch
    lr_gamma = 0.5
    num_epochs = 12
    batch_size = 1024 * 4
    log_interval = 1

    # Model
    model = Pinn(hidden_dims=hidden_dims)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=lr_step, gamma=lr_gamma
    )
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters: {num_params}")

    # Data
    data_path = Path("E:/donny/code/family/00/data/data.jsonl")
    train_data, test_data = get_dataset(data_path)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    # Training
    print("====== Training ======")
    print(f"# epochs: {num_epochs}")
    print(f"# examples: {len(train_data)}")
    print(f"batch size: {batch_size}")
    print(f"# steps: {len(train_loader)}")
    print(f"learning rate: {lr}")
    loss_history = []
    model.train()
    model.to(device)

    train_start_time = time()
    for ep in range(num_epochs):
        print(f"====== Epoch {ep} ======")
        for step, batch in enumerate(train_loader):
            inputs = {k: t.to(device) for k, t in batch.items()}

            # Forward
            outputs = model(**inputs)
            loss = outputs["loss"]
            loss_history.append(loss.item())

            # Backward
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if step % log_interval == 0:
                print(
                    {
                        "step": step,
                        "loss": round(loss.item(), 6),
                        "lr": round(optimizer.param_groups[0]["lr"], 4),
                        "time": round(time() - train_start_time, 1),
                    }
                )
        lr_scheduler.step()
        print(f"====== Epoch {ep} done ======")
        # Evaluate and save
        ckpt_dir = Path("result/pinn")
        ckpt_path = ckpt_dir / f"ckpt_{ep}.pt"
        torch.save(model.state_dict(), ckpt_path)
    print("====== Training done ======")

    # Testing
    print("====== Testing ======")
    model.eval()
    model.to(device)
    with torch.no_grad():
        for step, batch in enumerate(test_loader):
            batch = {key: t.to(device) for key, t in batch.items()}
            outputs = model(**batch)
            print(
                {
                    "step": step,
                    "loss": loss.item(),
                }
            )
    print("====== Testing done ======")


if __name__ == "__main__":
    main()
