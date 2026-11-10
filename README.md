# Bilinear Gating Unit (BGU)

Research code for **"Bilinear Gating Unit: Adaptive Expert-Gate Coupling through Bilinear Entanglement"**, an ESANN 2027 submission.

This repository contains PyTorch implementations of `BRGU`, `BMGU` and `BDGU`, alongside the `Maxout` and `MoE` baselines. It provides training and test pipelines for synthetic regression and Tiny-ImageNet classification.

## Repository structure

- [`units.py`](./units.py) - adaptive units, temperature scaling, and parameter-count helpers
- [`modules.py`](./modules.py) - shared seeding, early stopping, and normalization utilities
- [`synthetic/`](./synthetic/) - synthetic dataset, configuration files, and train/test scripts
- [`tiny-imagenet/`](./tiny-imagenet/) - Tiny-ImageNet wrapper, configuration files, and train/test scripts
- [`best_hyperparams.ipynb`](./best_hyperparams.ipynb) - optional helper for inspecting a training database
- [`requirements.txt`](./requirements.txt) - pinned Python dependencies

## Requirements and setup

Create a virtual environment with Python 3.13.

```bash
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

The train and test pipelines use a CUDA-backed multiprocessing scheduler and require at least one visible CUDA device.

## Running the benchmarks

Run commands from the repository root. Before training, review each benchmark's `config_train.yaml` and `config_test.yaml` to set the units, parameter budget, data options, learning rates, seeds, dropout, and run names.

### Synthetic regression

The synthetic benchmark evaluates ten functions on one-, two-, and three-dimensional inputs. The training configuration selects `overfitting` or `generalization` mode.

```bash
python synthetic/pipeline_synthetic_train.py
python synthetic/pipeline_synthetic_test.py
```

The test pipeline selects configurations using the mean recorded best loss for each unit and data setting, then runs the selected configurations with the seeds in `config_test.yaml`. The selection loss is training loss in `overfitting` mode and validation loss in `generalization` mode.

### Tiny-ImageNet classification

Tiny-ImageNet has 200 classes. The training pipeline creates a validation split from the training data and keeps the test split separate.

```bash
python tiny-imagenet/pipeline_tiny-imagenet_train.py
python tiny-imagenet/pipeline_tiny-imagenet_test.py
```

The test pipeline selects each unit's configuration using its mean best validation loss, then evaluates the selected configurations with the seeds in `config_test.yaml`.

### Configuration and results

The train scripts read `config_train.yaml`, test scripts read `config_test.yaml` and the corresponding training configuration. `training.name` sets the SQLite database filename, and `training.concurrent_gpu_jobs` controls the number of jobs allowed per GPU.

Databases are written to `synthetic/results/` or `tiny-imagenet/results/`. **The scripts delete an existing database at the configured path before starting a new run.** Back up results you want to keep or change `training.name` before rerunning.

## Results analysis

After a benchmark's test pipeline has completed, run its analysis script from the repository root:

```bash
python synthetic/pipeline_eval.py
python tiny-imagenet/pipeline_eval.py
```

Each script reads the test database named by `training.name` in that benchmark's `config_test.yaml` from its `results/` directory. It updates the database with derived analysis data and writes summary CSVs alongside it. CPU timings are collected in all cases, with GPU timing values being `NaN` when CUDA is unavailable. The optional [`best_hyperparams.ipynb`](./best_hyperparams.ipynb) helper requires its database path to be set manually.
