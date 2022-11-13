# PINN (Physics-Informed Neural Networks) on Navier-Stokes Equations

A faithful reimplementation of the [PINN](https://arxiv.org/abs/1711.10561) model on Navier-Stokes Equations using PyTorch. The model details is copied and translated from the [official implementation](https://github.com/maziarraissi/PINNs) of that paper.

## Requirements

- Python 3.6+
- PyTorch 1.0+

## Usage

### Data

The training code assumes that the data is placed in `../data/data.jsonl`, a file where each line (except for the first line) contains a JSON array of 6 floats: 

```jsonl
["t", "x", "y", "pressure", "u", "v"]
[2, 0.001, 0.0, 0.001698680166, 0.0, 0.0]
[2, 0.002, 0.0, 0.001695376845, 0.0, 0.0]
[2, 0.003, 0.0, 0.001699161309, 0.0, 0.0]
[2, 0.004, 0.0, 0.001711450528, 0.0, 0.0]
...
```

This will be loaded and split into training and test data by `main.py`.

### Training

Set the hyperparameters in `main.py` and run it. The model will be saved to `result/pinn` and the training log will be saved to `result/log`.

```bash
python main.py
```