# PINN (Physics-Informed Neural Networks) on Navier-Stokes Equations

A faithful reimplementation of the [PINN](https://arxiv.org/abs/1711.10561) model on Navier-Stokes Equations using PyTorch. The model details is copied and translated from the [official implementation](https://github.com/maziarraissi/PINNs) of that paper.

## Requirements

- Python 3.6+
- PyTorch 1.0+

## Model

I believe the official implementation of the PINN model is not very good, it converges too slowly, and is too narrow, resulting in very much wasted computational power of modern hardware. 

The original model mainly consists of a feed-forward network with 8 fully-connected layers with 20 neurons (amounts to 3072 weights). 

Our model instead consists of feed forward blocks, which is essentially the FFN part of the famous transformers model. The hyperparameters of the FFN blocks:

- Number of block: 8
- Hidden dimension: 128
- Intermediate dimension: 512
- Activation function: tanh 
- Dropout probability: 0.1

> Chose tanh because it is said to work better for problems revolving periodicity and high-order gradients.

This model has 1,054,468 parameters.

## Usage

### Data

### Use the data from the original PINN paper

This code now assumes the original data is placed at `../PINNs/main/Data/cylinder_nektar_wake.mat`, the function for loading data is `get_orig_dataset()` in `data.py`.

### Use custom data

Change to using `get_dataset(data_path)` from `data.py` to load your own data. This function assumes that the data is placed in `../data/data.jsonl`, a file where each line (except for the first line) contains a JSON array of 6 floats: 

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

## Code Architecture

- `data.py`: Data loading and preprocessing
- `main.py`: Training and testing
- `model.py`: PINN model (`Pinn`).
- `utils.py`: Utility functions
- `trainer.py`: A `Trainer` class, for more convenient training and testing

Right now, most hyperparameters are hard-coded in `Trainer` class (in `trainer.py`) and in the `Pinn` class (in `model.py`).
