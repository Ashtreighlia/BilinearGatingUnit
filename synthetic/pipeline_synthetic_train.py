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

torch.set_default_dtype(torch.float64)

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

data_modes = config["data"]["mode"]
data_types = config["data"]["type"]
data_dimensions = config["data"]["dimensions"]
data_splits = config["data"]["splits"]

criterions = config["training"]["criterions"]
optimizers = config["training"]["optimizers"]

global_data_resolution = config["data"]["resolution"]
global_max_epochs = config["training"]["max_epochs"]
global_seed = config["training"]["global_seed"]

concurrent_gpu_jobs = config["training"].get("concurrent_gpu_jobs", 1)

# Build the training configuration grid
runs = [
    {
        "unit_type": unit_type,
        "unit_params": unit_parameter,
        "data_resolution": global_data_resolution,
        "data_splits": data_splits,
        "data_mode": data_mode,
        "data_type": data_type,
        "data_dimension": data_dimension,
        "learning_rate": lr,
        "seed": seed,
        "dropout": dropout,
        "global_seed": global_seed,
        "max_epochs": global_max_epochs,
        "optimizer": optimizer,
        "criterion": criterion
    }
    for unit_type, unit_parameter, data_mode, data_type, data_dimension, lr, seed, dropout, optimizer, criterion in itertools.product(
        unit_types,
        unit_parameters,
        data_modes,
        data_types,
        data_dimensions,
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
        data_resolution INTEGER,
        data_mode TEXT,
        data_splits TEXT,
        data_type TEXT,
        data_dimension INTEGER,
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
            best_model BLOB,
            log BLOB,
            train_data BLOB,
            val_data BLOB,
            test_data BLOB,
            FOREIGN KEY(run_id) REFERENCES run_configs(id)
        )
''')

cur.execute('CREATE INDEX IF NOT EXISTS idx_run_results_run_id ON run_results(run_id)')

cur.execute('''
    CREATE VIEW IF NOT EXISTS run_summary AS
    SELECT 
        rc.id as run_id,
        rc.unit_type,
        rc.unit_params,
        rc.data_resolution,
        rc.data_mode,
        rc.data_splits,
        rc.data_type,
        rc.data_dimension,
        rc.learning_rate,
        rc.seed,
        rc.dropout,
        rc.global_seed,
        rc.max_epochs,
        rc.optimizer,
        rc.criterion,
        rr.best_loss,
        rr.duration,
        rr.epoch_stopped,
        rr.unit_sub_units,
        rr.unit_parameters
    FROM run_configs rc
    LEFT JOIN run_results rr ON rc.id = rr.run_id
''')

con.commit()

data_splits_str = str(data_splits)

rows = []
for i, run in tqdm.tqdm(enumerate(runs), total=len(runs), desc="Running configurations"):
    rows.append((
        run["unit_type"],
        run["unit_params"],
        run["data_resolution"],
        run["data_mode"],
        data_splits_str,
        run["data_type"],
        run["data_dimension"],
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
        unit_type, unit_params, data_resolution, data_mode, data_splits,
        data_type, data_dimension, learning_rate, seed, dropout,
        global_seed, max_epochs, optimizer, criterion
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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

# Run all configurations in parallel
parallel_training(cur_result, config_dicts, concurrent_gpu_jobs)

# %%
