from pathlib import Path
from time import time

import torch
from torch.utils.data import RandomSampler, DataLoader

from model import Pinn
from data import dump_json, PinnDataset


class Trainer:
    """Trainer for convenient training and testing"""

    def __init__(
        self,
        model: Pinn,
        output_dir: Path = None,
        lr: float = 0.001,
        num_epochs: int = 40,
        batch_size: int = 128,
    ):
        self.model = model

        # Hyperparameters
        self.lr = lr
        self.lr_step = 5  # Unit is epoch
        self.lr_gamma = 0.8
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.log_interval = 1
        self.samples_per_ep = 5000
        
        if output_dir is None:
            self.output_dir = Path(
                "result",
                "pinn-large-tanh",
                f"bs{batch_size}"
                f"-lr{lr}"
                f"-lrstep{self.lr_step}"
                f"-lrgamma{self.lr_gamma}"
                f"-epoch{self.num_epochs}",
            )
        else:
            self.output_dir = output_dir

        print(f"Output dir: {self.output_dir}")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        args = {}
        for attr in ["lr", "lr_step", "lr_gamma", "num_epochs", "batch_size"]:
            args[attr] = getattr(self, attr)
        dump_json(self.output_dir / "args.json", args)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=1, gamma=self.lr_gamma
        )

    def get_last_ckpt_dir(self) -> Path:
        ckpt_dirs = list(self.output_dir.glob("ckpt-*"))
        ckpt_dirs.sort(key=lambda x: int(x.name.split("-")[-1]))
        if len(ckpt_dirs) == 0:
            return None
        return ckpt_dirs[-1]

    def train(self, train_data: PinnDataset, do_resume: bool = True):
        model = self.model
        device = self.device

        sampler = RandomSampler(
            train_data,
            replacement=True,
            num_samples=self.samples_per_ep,
        )
        train_loader = DataLoader(
            train_data, batch_size=self.batch_size, sampler=sampler
        )

        print("====== Training ======")
        print(f'device is "{device}"')
        print(f"# epochs: {self.num_epochs}")
        print(f"# examples: {len(train_data)}")
        print(f"# samples used per epoch: {self.samples_per_ep}")
        print(f"batch size: {self.batch_size}")
        print(f"# steps: {len(train_loader)}")
        self.loss_history = []
        model.train()
        model.to(device)

        # Resume
        last_ckpt_dir = self.get_last_ckpt_dir()
        if do_resume and last_ckpt_dir is not None:
            print(f"Resuming from {last_ckpt_dir}")
            self.load_ckpt(last_ckpt_dir)
            ep = int(last_ckpt_dir.name.split("-")[-1]) + 1
        else:
            ep = 0

        train_start_time = time()
        while ep < self.num_epochs:
            print(f"====== Epoch {ep} ======")
            for step, batch in enumerate(train_loader):
                inputs = {k: t.to(device) for k, t in batch.items()}

                # Forward
                outputs = model(**inputs)
                loss = outputs["loss"]
                self.loss_history.append(loss.item())

                # Backward
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

                if step % self.log_interval == 0:
                    losses = outputs["losses"]
                    print(
                        {
                            "step": step,
                            "loss": round(loss.item(), 6),
                            "lr": round(
                                self.optimizer.param_groups[0]["lr"], 4
                            ),
                            "lambda1": round(self.model.lambda1.item(), 4),
                            "lambda2": round(self.model.lambda2.item(), 4),
                            "u_loss": round(losses["u_loss"].item(), 6),
                            "v_loss": round(losses["v_loss"].item(), 6),
                            "f_u_loss": round(losses["f_u_loss"].item(), 6),
                            "f_v_loss": round(losses["f_v_loss"].item(), 6),
                            "time": round(time() - train_start_time, 1),
                        }
                    )
            self.lr_scheduler.step()
            self.checkpoint(ep)
            print(f"====== Epoch {ep} done ======")
            ep += 1
        print("====== Training done ======")

    def checkpoint(self, ep: int):
        """
        Dump checkpoint (model, optimizer, lr_scheduler) to "ckpt-{ep}" in
        the `output_dir`,

        and dump `self.loss_history` to "loss_history.json" in the
        `ckpt_dir`, and clear `self.loss_history`.
        """
        # Evaluate and save
        ckpt_dir = self.output_dir / f"ckpt-{ep}"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        print(f"Checkpointing to {ckpt_dir}")
        torch.save(self.model.state_dict(), ckpt_dir / "ckpt.pt")
        torch.save(self.optimizer.state_dict(), ckpt_dir / "optimizer.pt")
        torch.save(
            self.lr_scheduler.state_dict(), ckpt_dir / "lr_scheduler.pt"
        )
        dump_json(ckpt_dir / "loss_history.json", self.loss_history)
        self.loss_history = []

    def load_ckpt(self, ckpt_dir: Path):
        print(f'Loading checkpoint from "{ckpt_dir}"')
        self.model.load_state_dict(torch.load(ckpt_dir / "ckpt.pt"))
        self.optimizer.load_state_dict(torch.load(ckpt_dir / "optimizer.pt"))
        self.lr_scheduler.load_state_dict(
            torch.load(ckpt_dir / "lr_scheduler.pt")
        )

    def predict(self, test_data: PinnDataset) -> dict:
        batch_size = self.batch_size * 32
        test_loader = DataLoader(
            test_data, batch_size=batch_size, shuffle=False
        )
        print("====== Testing ======")
        print(f"# examples: {len(test_data)}")
        print(f"batch size: {batch_size}")
        print(f"# steps: {len(test_loader)}")

        self.model.to(self.device)
        self.model.train()  # We need gradient to predict
        all_preds = []
        all_losses = []
        for step, batch in enumerate(test_loader):
            batch = {key: t.to(self.device) for key, t in batch.items()}
            outputs = self.model(**batch)
            all_losses.append(outputs["loss"].item())
            all_preds.append(outputs["preds"])
        print("====== Testing done ======")
        all_preds = torch.cat(all_preds, 0)
        loss = sum(all_losses) / len(all_losses)
        return {
            "loss": loss,
            "preds": all_preds,
        }
