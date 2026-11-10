#%%
import csv
import os
import pickle
import random as _random
import sqlite3
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import tqdm
import yaml

BENCH_DIR = Path(__file__).resolve().parent
ROOT = BENCH_DIR.parent
sys.path.insert(0, str(ROOT))
import units

#%%
RESULT_DIR = BENCH_DIR / "results"
with open(BENCH_DIR / "config_test.yaml") as f:
    test_config = yaml.safe_load(f)
test_db_name = test_config["training"]["name"] + ".db"
test_db_path = RESULT_DIR / test_db_name
con = sqlite3.connect(test_db_path)
cur = con.cursor()

#%%
# Extract evaluation metrics
cur.execute('DROP TABLE IF EXISTS run_metrics')
cur.execute('''
    CREATE TABLE run_metrics (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        run_id INTEGER,
        epoch INTEGER,
        mae REAL,
        std REAL,
        rmse REAL,
        FOREIGN KEY(run_id) REFERENCES run_configs(id)
    )
''')
all_rows = cur.execute('SELECT * FROM run_blobs').fetchall()
metrics_data = []
for row in tqdm.tqdm(all_rows, desc="Processing runs"):
    data = pickle.loads(row[6] if row[6] is not None else row[4])
    targets, outputs = data[1], data[2]
    metrics_data.append((
        row[1], 0,
        torch.mean(torch.abs(targets - outputs)).item(),
        torch.std(targets - outputs).item(),
        torch.sqrt(torch.mean((targets - outputs) ** 2)).item(),
    ))
cur.executemany('INSERT INTO run_metrics (run_id, epoch, mae, std, rmse) VALUES (?, ?, ?, ?, ?)', metrics_data)
con.commit()

#%%
# Benchmark unit costs
cur.execute("DROP TABLE IF EXISTS compute_costs")
cur.execute("""
    CREATE TABLE compute_costs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        unit_type TEXT,
        in_features INTEGER,
        num_sub_units INTEGER,
        time_forward_gpu REAL,
        time_backward_gpu REAL,
        time_forward_cpu REAL,
        time_backward_cpu REAL,
        UNIQUE(unit_type, in_features, num_sub_units)
    )
""")


def initialize_unit(unit_type, in_features, num_sub_units):
    unit_class = getattr(units, unit_type)
    return unit_class(
        in_features=in_features,
        out_features=1,
        sub_units=num_sub_units,
        dropout=0.0,
        seed=None,
    )

def benchmark_units_interleaved(unit_configs, device, iterations=1000, warmup=100, batch_size=1000):
    is_cuda = device == "cuda"

    if is_cuda and not torch.cuda.is_available():
        return {key: (np.nan, np.nan) for key, _, _ in unit_configs}

    prepared = []
    for key, unit, in_features in unit_configs:
        unit = unit.to(device)
        unit.eval()

        inp = torch.randn(batch_size, in_features, device=device)
        prepared.append((key, unit, inp))

    # Measure inference latency without gradients
    with torch.inference_mode():
        for _ in range(warmup):
            for _, unit, inp in prepared:
                unit(inp)
        fwd_times = {key: 0.0 for key, _, _ in prepared}

        for _ in range(iterations):
            # Randomize order to reduce timing bias
            for key, unit, inp in _random.sample(prepared, len(prepared)):
                if is_cuda:
                    torch.cuda.synchronize()

                t0 = time.perf_counter()
                unit(inp)

                if is_cuda:
                    torch.cuda.synchronize()

                fwd_times[key] += time.perf_counter() - t0

    # Measure backward latency only
    for _, unit, _ in prepared:
        unit.train()

    # Warm up backward passes
    for _ in range(warmup):
        for _, unit, inp in prepared:
            unit.zero_grad(set_to_none=True)
            out = unit(inp).sum()
            out.backward()

    if is_cuda:
        torch.cuda.synchronize()

    bwd_times = {key: 0.0 for key, _, _ in prepared}

    for _ in range(iterations):
        # Randomize order to reduce timing bias
        for key, unit, inp in _random.sample(prepared, len(prepared)):
            unit.zero_grad(set_to_none=True)

            # Exclude graph construction from the timed region
            out = unit(inp).sum()

            if is_cuda:
                torch.cuda.synchronize()

            t0 = time.perf_counter()
            out.backward()

            if is_cuda:
                torch.cuda.synchronize()

            bwd_times[key] += time.perf_counter() - t0

    # Return average time per batch
    return {
        key: (
            fwd_times[key] / iterations,
            bwd_times[key] / iterations,
        )
        for key, _, _ in prepared
    }

unique_configs = cur.execute("""
    SELECT DISTINCT rc.unit_type, rc.data_dimension, rr.unit_sub_units
    FROM run_configs rc
    JOIN run_results rr ON rc.id = rr.run_id
    ORDER BY rc.unit_type, rc.data_dimension, rr.unit_sub_units
""").fetchall()

all_unit_configs_cpu = []
all_unit_configs_gpu = []

for unit_type, in_features, num_sub_units in unique_configs:
    key = (unit_type, in_features, num_sub_units)

    all_unit_configs_cpu.append(
        (key, initialize_unit(unit_type, in_features, num_sub_units), in_features)
    )

    all_unit_configs_gpu.append(
        (key, initialize_unit(unit_type, in_features, num_sub_units), in_features)
    )

cpu_results = benchmark_units_interleaved(all_unit_configs_cpu, "cpu")
gpu_results = benchmark_units_interleaved(all_unit_configs_gpu, "cuda")

for unit_type, in_features, num_sub_units in tqdm.tqdm(unique_configs, desc="Saving results"):
    key = (unit_type, in_features, num_sub_units)

    time_fwd_cpu, time_bwd_cpu = cpu_results[key]
    time_fwd_gpu, time_bwd_gpu = gpu_results[key]

    cur.execute("""
        INSERT OR REPLACE INTO compute_costs 
        (
            unit_type,
            in_features,
            num_sub_units,
            time_forward_gpu,
            time_backward_gpu,
            time_forward_cpu,
            time_backward_cpu
        )
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (
        unit_type,
        in_features,
        num_sub_units,
        time_fwd_gpu,
        time_bwd_gpu,
        time_fwd_cpu,
        time_bwd_cpu,
    ))

con.commit()

#%%
# Summarize metrics and write ESANN tables
query = """
SELECT 
    rc.unit_type, rc.unit_params, rc.data_resolution, rc.data_mode, 
    rc.data_splits, rc.data_type, rc.data_dimension, rc.learning_rate, 
    rc.seed, rc.dropout, rc.global_seed, rc.max_epochs, rc.optimizer, rc.criterion,
    rm.mae, rm.std, rm.rmse,
    rr.unit_parameters, rr.duration,
    cc.time_forward_gpu, cc.time_backward_gpu, cc.time_forward_cpu, cc.time_backward_cpu
FROM run_configs rc
JOIN run_results rr ON rc.id = rr.run_id
JOIN run_metrics rm ON rc.id = rm.run_id
JOIN compute_costs cc ON (
    rc.unit_type = cc.unit_type AND
    rc.data_dimension = cc.in_features AND
    rr.unit_sub_units = cc.num_sub_units
)
"""
df = pd.read_sql_query(query, con)

EPS = 1e-12
for c in ["mae", "time_forward_cpu", "time_backward_cpu", "time_forward_gpu", "time_backward_gpu"]:
    df.loc[df[c] < EPS, c] = np.nan

summary = (
    df.groupby(["data_mode", "unit_type"])
      .agg(
          mae_mean=("mae", "mean"),
          mae_std=("mae", "std"),

          time_cpu_fwd_mean=("time_forward_cpu", "mean"),
          time_cpu_fwd_std=("time_forward_cpu", "std"),
          time_cpu_bwd_mean=("time_backward_cpu", "mean"),
          time_cpu_bwd_std=("time_backward_cpu", "std"),

          time_gpu_fwd_mean=("time_forward_gpu", "mean"),
          time_gpu_fwd_std=("time_forward_gpu", "std"),
          time_gpu_bwd_mean=("time_backward_gpu", "mean"),
          time_gpu_bwd_std=("time_backward_gpu", "std"),
      )
      .reset_index()
)

summary["eff_cpu_fwd"] = 1.0 / (summary["mae_mean"] * summary["time_cpu_fwd_mean"] + 1e-6)
summary["eff_cpu_bwd"] = 1.0 / (summary["mae_mean"] * summary["time_cpu_bwd_mean"] + 1e-6)
summary["eff_gpu_fwd"] = 1.0 / (summary["mae_mean"] * summary["time_gpu_fwd_mean"] + 1e-6)
summary["eff_gpu_bwd"] = 1.0 / (summary["mae_mean"] * summary["time_gpu_bwd_mean"] + 1e-6)

def add_delta_vs_moe(summary, value_col, out_col):
    summary[out_col] = np.nan

    for mode in summary["data_mode"].unique():
        mask_mode = summary["data_mode"] == mode
        moe = summary.loc[mask_mode & (summary["unit_type"] == "MoE"), value_col]

        assert len(moe) == 1, f"Expected exactly one MoE row for mode={mode}"
        moe_val = moe.iloc[0]

        summary.loc[mask_mode, out_col] = (
            summary.loc[mask_mode, value_col] / moe_val - 1.0
        ) * 100.0

    summary.loc[summary["unit_type"] == "MoE", out_col] = 0.0

add_delta_vs_moe(summary, "mae_mean", "delta_mae_vs_moe")
add_delta_vs_moe(summary, "time_cpu_fwd_mean", "delta_time_cpu_fwd_vs_moe")
add_delta_vs_moe(summary, "time_cpu_bwd_mean", "delta_time_cpu_bwd_vs_moe")
add_delta_vs_moe(summary, "time_gpu_fwd_mean", "delta_time_gpu_fwd_vs_moe")
add_delta_vs_moe(summary, "time_gpu_bwd_mean", "delta_time_gpu_bwd_vs_moe")

add_delta_vs_moe(summary, "eff_cpu_fwd", "delta_eff_cpu_fwd_vs_moe")
add_delta_vs_moe(summary, "eff_cpu_bwd", "delta_eff_cpu_bwd_vs_moe")
add_delta_vs_moe(summary, "eff_gpu_fwd", "delta_eff_gpu_fwd_vs_moe")
add_delta_vs_moe(summary, "eff_gpu_bwd", "delta_eff_gpu_bwd_vs_moe")

summary["mae_mean_std"] = summary.apply(
    lambda row: f"{row['mae_mean']:.2f}±{row['mae_std']:.2f}" if pd.notnull(row['mae_mean']) and pd.notnull(row['mae_std']) else np.nan,
    axis=1
)

def get_raw(data, unit, mode, cols):
    sub = data[(data["unit_type"] == unit) & (data["data_mode"] == mode)]
    if len(sub) == 0:
        return [np.nan] * len(cols)
    return sub[cols].values[0].tolist()


def write_csv(path, header, rows):
    os.makedirs(path.parent, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        
        formatted_rows = []
        for row in rows:
            formatted_rows.append([
                f"{val:.2f}" if isinstance(val, (int, float)) and pd.notnull(val) else val for val in row
            ])
        writer.writerows(formatted_rows)

SYN_OVER_OUT_FILE = RESULT_DIR / "generated_synthetic_overfitting_table.csv"
SYN_GEN_OUT_FILE = RESULT_DIR / "generated_synthetic_generalization_table.csv"

ORDER = ["BDGU", "BMGU", "BRGU", "MoE", "Maxout"]

headers = [
    "unit_type",
    "mae_mean",
    "mae_std",
    "mae_mean_std",
    "delta_mae_vs_moe",
    "eff_cpu_fwd",
    "delta_eff_cpu_fwd_vs_moe",
    "eff_cpu_bwd",
    "delta_eff_cpu_bwd_vs_moe",
    "eff_gpu_fwd",
    "delta_eff_gpu_fwd_vs_moe",
    "eff_gpu_bwd",
    "delta_eff_gpu_bwd_vs_moe"
]

row_cols = [
    "mae_mean", "mae_std", "mae_mean_std", "delta_mae_vs_moe",
    "eff_cpu_fwd", "delta_eff_cpu_fwd_vs_moe",
    "eff_cpu_bwd", "delta_eff_cpu_bwd_vs_moe",
    "eff_gpu_fwd", "delta_eff_gpu_fwd_vs_moe",
    "eff_gpu_bwd", "delta_eff_gpu_bwd_vs_moe"
]

over_rows, gen_rows = [], []
for unit in ORDER:
    ov_vals = get_raw(summary, unit, "overfitting", row_cols)
    gen_vals = get_raw(summary, unit, "generalization", row_cols)
    
    over_rows.append([unit] + ov_vals)
    gen_rows.append([unit] + gen_vals)

write_csv(SYN_OVER_OUT_FILE, headers, over_rows)
write_csv(SYN_GEN_OUT_FILE, headers, gen_rows)
print("Saved generalization and overfitting CSV Tables with stable aggregate deltas!")

#%%
# Save per-unit parameter summaries to CSV
MAX_PARAMS = 32

query = """
SELECT
    rc.unit_type,
    rr.unit_parameters
FROM run_configs rc
JOIN run_results rr ON rc.id = rr.run_id
"""
param_df = pd.read_sql_query(query, con)

if param_df.empty:
    raise RuntimeError("No parameter rows found in run_configs/run_results.")

observed_max = int(param_df["unit_parameters"].max())
if observed_max != MAX_PARAMS:
    print(f"Warning: expected max={MAX_PARAMS}, observed max={observed_max}")

summary = (
    param_df.groupby("unit_type", as_index=False)
    .agg(
        min_unit_parameters=("unit_parameters", "min"),
        max_unit_parameters=("unit_parameters", "max"),
        avg_unit_parameters=("unit_parameters", "mean"),
        std_unit_parameters=("unit_parameters", "std"),
    )
)

summary["avg_rel_to_max"] = (summary["avg_unit_parameters"] / MAX_PARAMS) * 100.0

summary = summary[[
    "unit_type",
    "min_unit_parameters",
    "max_unit_parameters",
    "avg_unit_parameters",
    "std_unit_parameters",
    "avg_rel_to_max",
]].sort_values("unit_type")

PARAM_SUMMARY_OUT_FILE = RESULT_DIR / "generated_synthetic_parameter_summary_table.csv"
summary.to_csv(PARAM_SUMMARY_OUT_FILE, index=False, float_format="%.6f")

print(f"Saved parameter summary CSV: {PARAM_SUMMARY_OUT_FILE}")
print(summary)
con.close()