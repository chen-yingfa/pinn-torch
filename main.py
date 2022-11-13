from pathlib import Path
import random

import numpy as np
import torch

from model import Pinn
from data import get_dataset
from trainer import Trainer


def process_test_result(
    test_data: torch.Tensor,
    loss: float,
    preds: torch.Tensor,
    lambda1: int,
    lambda2: int,
):
    p = test_data[:, 3]
    u = test_data[:, 4]
    v = test_data[:, 5]
    p_pred = preds[:, 0]
    u_pred = preds[:, 1]
    v_pred = preds[:, 2]

    # Error
    err_u = np.linalg.norm(u - u_pred, 2) / np.linalg.norm(u, 2)
    err_v = np.linalg.norm(v - v_pred, 2) / np.linalg.norm(v, 2)
    err_p = np.linalg.norm(p - p_pred, 2) / np.linalg.norm(p, 2)

    err_lambda1 = np.abs(lambda1 - 1.0)
    err_lambda2 = np.abs(lambda2 - 0.01) / 0.01

    print(f"Error in velocity: {err_u:.2e}, {err_v:.2e}")
    print(f"Error in pressure: {err_p:.2e}")
    print(f"Error in lambda 1: {err_lambda1:.2f}")
    print(f"Error in lambda 2: {err_lambda2:.2f}")

    # Plot
    pass


def main():
    torch.random.manual_seed(0)
    random.seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Model
    hidden_dims = [20] * 8
    model = Pinn(hidden_dims=hidden_dims)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters: {num_params}")

    # Data
    data_path = Path("E:/donny/code/family/00/data/data.jsonl")
    train_data, test_data = get_dataset(data_path)

    trainer = Trainer(model)
    trainer.train(train_data)
    outputs = trainer.predict(test_data)

    lambda1 = trainer.model.lambda1.item().cpu().detach()
    lambda2 = trainer.model.lambda2.item().cpu().detach()
    print(lambda1)
    print(lambda2)
    loss = outputs["loss"]
    preds = outputs["preds"]
    process_test_result(test_data, loss, preds, lambda1, lambda2)


if __name__ == "__main__":
    main()
