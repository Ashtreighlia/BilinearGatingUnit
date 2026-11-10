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


def main():
    with open(BENCH_DIR / "config_test.yaml", "r") as f:
        config_test = yaml.safe_load(f)

    with open(BENCH_DIR / "config_train.yaml", "r") as f:
        config_train = yaml.safe_load(f)

    # Read training results to select the best hyperparameters
    train_db_path = str(BENCH_DIR / "results" / (config_train["training"]["name"] + ".db"))
    if not os.path.exists(train_db_path):
        print(f"Error: Training database not found at {train_db_path}")
        return

    print(f"Reading training results from {train_db_path}")
    con_train = sqlite3.connect(train_db_path)
    cur_train = con_train.cursor()

    data_modes = config_train["data"]["mode"]
    data_types = config_train["data"]["type"]
    data_dimensions = config_train["data"]["dimensions"]
    
    test_units = config_test["units"]["types"]
    best_configs = []

    # Select the lowest-loss run for each unit and data setting
    for unit_type, data_mode, data_type, data_dimension in itertools.product(test_units, data_modes, data_types, data_dimensions):
        cur_train.execute('''
            SELECT c.learning_rate, c.dropout, c.unit_params, 
                   c.data_resolution, c.data_splits,
                   c.optimizer, c.criterion, AVG(r.best_loss) as avg_loss
            FROM run_configs c
            JOIN run_results r ON c.id = r.run_id
            WHERE c.unit_type = ? AND c.data_mode = ? AND c.data_type = ? AND c.data_dimension = ?
            GROUP BY c.learning_rate, c.dropout, c.unit_params, c.data_resolution, c.data_splits, c.optimizer, c.criterion
            ORDER BY avg_loss ASC
            LIMIT 1
        ''', (unit_type, data_mode, data_type, data_dimension))
        
        row = cur_train.fetchone()
        if row:
            best_configs.append({
                "unit_type": unit_type,
                "data_mode": data_mode,
                "data_type": data_type,
                "data_dimension": data_dimension,
                "learning_rate": row[0],
                "dropout": row[1],
                "unit_params": row[2],
                "data_resolution": row[3],
                "data_splits": row[4],
                "optimizer": row[5],
                "criterion": row[6]
            })
            print(f"Found best config for {unit_type} ({data_mode}, {data_type}, dim {data_dimension}): LR={row[0]}, Dropout={row[1]}")
        else:
            print(f"Warning: No training runs found for unit {unit_type} ({data_mode}, {data_type}, dim {data_dimension}). Using default.")
            best_configs.append({
                "unit_type": unit_type,
                "data_mode": data_mode,
                "data_type": data_type,
                "data_dimension": data_dimension,
                "learning_rate": config_train["grid_search"]["lr"]["min"],
                "dropout": config_train["grid_search"]["dropout"]["min"],
                "unit_params": config_train["units"]["parameters"][0],
                "data_resolution": config_train["data"]["resolution"],
                "data_splits": str(config_train["data"]["splits"]),
                "optimizer": config_train["training"]["optimizers"][0],
                "criterion": config_train["training"]["criterions"][0]
            })

    con_train.close()

    seeds_config = config_test["grid_search"]["seeds"]
    seeds = np.arange(
        seeds_config["start"],
        seeds_config["start"] + seeds_config["steps"]
    ).tolist()

    global_max_epochs = config_train["training"]["max_epochs"]
    global_seed = config_train["training"]["global_seed"]
    concurrent_gpu_jobs = config_train["training"].get("concurrent_gpu_jobs", 1)

    # Build test runs from the selected settings and test seeds
    runs = []
    for bc in best_configs:
        for seed in seeds:
            runs.append({
                "unit_type": bc["unit_type"],
                "unit_params": bc["unit_params"],
                "data_resolution": bc["data_resolution"],
                "data_splits": str(bc["data_splits"]),
                "data_mode": bc["data_mode"],
                "data_type": bc["data_type"],
                "data_dimension": bc["data_dimension"],
                "learning_rate": bc["learning_rate"],
                "seed": seed,
                "dropout": bc["dropout"],
                "global_seed": global_seed,
                "max_epochs": global_max_epochs,
                "optimizer": bc["optimizer"],
                "criterion": bc["criterion"]
            })

    if not os.path.exists(BENCH_DIR / "results"):
        os.makedirs(BENCH_DIR / "results")

    # Replace any existing test database
    db_name = config_test["training"]["name"] + ".db"
    path = str(BENCH_DIR / "results" / db_name)
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

    con.commit()

    rows = []
    for run in runs:
        rows.append((
            run["unit_type"],
            run["unit_params"],
            run["data_resolution"],
            run["data_mode"],
            str(run["data_splits"]),
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

    cur.execute("SELECT * FROM run_configs")
    configs = cur.fetchall()

    config_dicts = [dict(zip(keys, row)) for row in configs]

    cur_result = con.cursor()

    print(f"Starting evaluations for {len(config_dicts)} test configurations...")

    # Evaluate the selected configurations in parallel
    parallel_training(cur_result, config_dicts, concurrent_gpu_jobs)

    con.close()

if __name__ == "__main__":
    main()
