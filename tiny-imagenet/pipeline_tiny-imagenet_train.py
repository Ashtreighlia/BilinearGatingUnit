# %%
import os
import random
import time
import yaml
import itertools
import sqlite3
import sys
from pathlib import Path

import torch
import numpy as np
import tqdm

torch.set_default_dtype(torch.float32)

from pipeline_multiprocess import parallel_training

BENCH_DIR = Path(__file__).resolve().parent

with open(BENCH_DIR / "config_train.yaml", "r") as f:
    config = yaml.safe_load(f)

unit_combinations = []
unit_types = config["units"]["types"]
unit_parameters = config["units"]["parameters"]

grid_search_config = config["grid_search"]
lr_config = grid_search_config["lr"]
seeds_config = grid_search_config["seeds"]
dropout_config = grid_search_config["dropout"]

learning_rates = np.logspace(
    np.log10(lr_config["min"]),
    np.log10(lr_config["max"]),
    lr_config["steps"]
).tolist()

seeds = np.arange(
    seeds_config["start"],
    seeds_config["start"] + seeds_config["steps"]
).tolist()

dropouts = np.linspace(
    dropout_config["min"],
    dropout_config["max"],
    dropout_config["steps"]
).tolist()

data_split = config["data"]["split"]
data_batch_size = config["data"]["batch_size"]

criterions = config["training"]["criterions"]
optimizers = config["training"]["optimizers"]

global_max_epochs = config["training"]["max_epochs"]
global_seed = config["training"]["global_seed"]

concurrent_gpu_jobs = config["training"].get("concurrent_gpu_jobs", 1)

# Build the training configuration grid
runs = [
    {
        "unit_type": unit_type,
        "unit_params": unit_parameter,
        "data_split": data_split,
        "data_batch_size": data_batch_size,
        "learning_rate": lr,
        "seed": seed,
        "dropout": dropout,
        "global_seed": global_seed,
        "max_epochs": global_max_epochs,
        "optimizer": optimizer,
        "criterion": criterion
    }
    for unit_type, unit_parameter, data_split, data_batch_size, lr, seed, dropout, optimizer, criterion in itertools.product(
        unit_types,
        unit_parameters,
        data_split,
        data_batch_size,
        learning_rates,
        seeds,
        dropouts,
        optimizers,
        criterions
    )
]

if not os.path.exists(BENCH_DIR / "results"):
    os.makedirs(BENCH_DIR / "results")

# Replace any existing training database
path = str(BENCH_DIR / "results" / (config["training"]["name"] + ".db"))
if os.path.exists(path):
    os.remove(path)
con = sqlite3.connect(path)
cur = con.cursor()

cur.execute('''
    CREATE TABLE IF NOT EXISTS run_configs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        unit_type TEXT,
        unit_params INTEGER,
        data_split REAL,
        data_batch_size INTEGER,
        learning_rate REAL,
        seed INTEGER,
        dropout REAL,
        global_seed INTEGER,
        max_epochs INTEGER,
        optimizer TEXT,
        criterion TEXT
    )
''')

cur.execute('''
    CREATE TABLE IF NOT EXISTS run_results (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        run_id INTEGER,
        best_loss REAL,
        test_loss REAL,
        test_accuracy REAL,
        test_f1_score REAL,
        test_auroc REAL,
        duration REAL,
        epoch_stopped INTEGER,
        unit_sub_units INTEGER,
        unit_parameters INTEGER,
        FOREIGN KEY(run_id) REFERENCES run_configs(id)
    )
''')

cur.execute('''
        CREATE TABLE IF NOT EXISTS run_blobs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id INTEGER,
            best_encoder BLOB,
            best_model BLOB,
            log BLOB,
            FOREIGN KEY(run_id) REFERENCES run_configs(id)
        )
''')


con.commit()

rows = []
for i, run in tqdm.tqdm(enumerate(runs), total=len(runs), desc="Running configurations"):
    rows.append((
        run["unit_type"],
        run["unit_params"],
        run["data_split"],
        run["data_batch_size"],
        run["learning_rate"],
        run["seed"],
        run["dropout"],
        run["global_seed"],
        run["max_epochs"],
        run["optimizer"],
        run["criterion"]
    ))

cur.execute('BEGIN')
cur.executemany('''
    INSERT INTO run_configs (
        unit_type, unit_params, data_split,
        data_batch_size, learning_rate, seed, dropout,
        global_seed, max_epochs, optimizer, criterion
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
''', rows)
cur.execute('COMMIT')

con.commit()

cur.execute("SELECT * FROM run_configs LIMIT 0")
keys = [desc[0] for desc in cur.description]
print("run_configs columns:", keys)

cur.execute("SELECT * FROM run_configs")
configs = cur.fetchall()

config_dicts = [dict(zip(keys, row)) for row in configs]

cur_result = con.cursor()

# for cfg in tqdm.tqdm(config_dicts, desc="Training runs"):
#     print(cfg)

parallel_training(cur_result, config_dicts, concurrent_gpu_jobs)