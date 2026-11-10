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

    test_units = config_test["units"]["types"]
    best_configs = {}

    for unit_type in test_units:
        # Select the lowest-loss configuration for each unit type
        cur_train.execute('''
            SELECT c.learning_rate, c.dropout, c.unit_params, c.data_split, 
                   c.data_batch_size, c.optimizer, c.criterion, AVG(r.best_loss) as avg_loss
            FROM run_configs c
            JOIN run_results r ON c.id = r.run_id
            WHERE c.unit_type = ?
            GROUP BY c.learning_rate, c.dropout, c.unit_params, c.data_split, c.data_batch_size, c.optimizer, c.criterion
            ORDER BY avg_loss ASC
            LIMIT 1
        ''', (unit_type,))
        
        row = cur_train.fetchone()
        if row:
            best_configs[unit_type] = {
                "learning_rate": row[0],
                "dropout": row[1],
                "unit_params": row[2],
                "data_split": row[3],
                "data_batch_size": row[4],
                "optimizer": row[5],
                "criterion": row[6]
            }
            print(f"Found best config for {unit_type}: LR={row[0]}, Dropout={row[1]}, Avg Loss={row[7]:.4f}")
        else:
            print(f"Warning: No training runs found for unit {unit_type}. Taking default from config_train.")
            best_configs[unit_type] = {
                "learning_rate": config_train["grid_search"]["lr"]["min"],
                "dropout": config_train["grid_search"]["dropout"]["min"],
                "unit_params": config_train["units"]["parameters"][0],
                "data_split": config_train["data"]["split"][0],
                "data_batch_size": config_train["data"]["batch_size"][0],
                "optimizer": config_train["training"]["optimizers"][0],
                "criterion": config_train["training"]["criterions"][0]
            }

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
    for unit_type in test_units:
        bc = best_configs[unit_type]
        for seed in seeds:
            runs.append({
                "unit_type": unit_type,
                "unit_params": bc["unit_params"],
                "data_split": bc["data_split"],
                "data_batch_size": bc["data_batch_size"],
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
    cur.execute('CREATE INDEX IF NOT EXISTS idx_run_results_run_id ON run_results(run_id)')
    con.commit()

    rows = []
    for run in runs:
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

    cur.execute("SELECT * FROM run_configs")
    configs = cur.fetchall()

    config_dicts = [dict(zip(keys, row)) for row in configs]

    cur_result = con.cursor()

    print(f"Starting evaluations for {len(config_dicts)} test configurations...")
    parallel_training(cur_result, config_dicts, concurrent_gpu_jobs)

    con.close()

if __name__ == "__main__":
    main()
