#%%
import csv
import inspect
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
    if hasattr(units, "parse_unit_type"):
        base_type, scaling_mode = units.parse_unit_type(unit_type)
    else:
        base_type, scaling_mode = unit_type, "constant"
    try:
        unit_class = getattr(units, base_type)
    except AttributeError as exc:
        raise ValueError(f"Unknown unit type: {unit_type}") from exc
    kwargs = {
        "in_features": in_features,
        "out_features": in_features,
        "sub_units": num_sub_units,
        "dropout": 0.0,
        "seed": None,
    }
    if "scaling_mode" in inspect.signature(unit_class.__init__).parameters:
        kwargs["scaling_mode"] = scaling_mode
    return unit_class(**kwargs)


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

        if is_cuda:
            torch.cuda.synchronize()

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
            out = unit(inp).sum()
            if is_cuda:
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            out.backward()
            if is_cuda:
                torch.cuda.synchronize()
            bwd_times[key] += time.perf_counter() - t0

    return {
        key: (
            fwd_times[key] / iterations,
            bwd_times[key] / iterations,
        )
        for key, _, _ in prepared
    }


unique_configs = cur.execute("""
    SELECT DISTINCT rc.unit_type, 128 AS in_features, rr.unit_sub_units
    FROM run_configs rc
    JOIN run_results rr ON rc.id = rr.run_id
    ORDER BY rc.unit_type, rr.unit_sub_units
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

print("  Benchmarking on CPU (interleaved)...")
cpu_results = benchmark_units_interleaved(all_unit_configs_cpu, "cpu")
print("  Benchmarking on GPU (interleaved)...")
gpu_results = benchmark_units_interleaved(all_unit_configs_gpu, "cuda")

for unit_type, in_features, num_sub_units in tqdm.tqdm(unique_configs, desc="Saving results"):
    key = (unit_type, in_features, num_sub_units)
    time_fwd_cpu, time_bwd_cpu = cpu_results[key]
    time_fwd_gpu, time_bwd_gpu = gpu_results[key]

    cur.execute("""
        INSERT OR REPLACE INTO compute_costs 
        (unit_type, in_features, num_sub_units, time_forward_gpu, time_backward_gpu, time_forward_cpu, time_backward_cpu)
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
def write_csv(path, header, rows, col_formats=None):
    os.makedirs(path.parent, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        formatted_rows = []
        for row in rows:
            formatted_rows.append([
                format(val, (col_formats or {}).get(col, ".4f")) if isinstance(val, float) and pd.notnull(val) else val
                for col, val in zip(header, row)
            ])
        writer.writerows(formatted_rows)


query = """
SELECT 
    rc.unit_type, rc.unit_params, rc.data_split, rc.data_batch_size,
    rc.learning_rate, rc.seed, rc.dropout, rc.global_seed, rc.max_epochs, rc.optimizer, rc.criterion,
    rr.best_loss, rr.test_loss, rr.test_accuracy, rr.test_f1_score, rr.test_auroc, rr.duration, 
    rr.epoch_stopped, rr.unit_sub_units, rr.unit_parameters,
    cc.time_forward_gpu, cc.time_backward_gpu, cc.time_forward_cpu, cc.time_backward_cpu
FROM run_configs rc
JOIN run_results rr ON rc.id = rr.run_id
JOIN compute_costs cc ON rc.unit_type = cc.unit_type AND rr.unit_sub_units = cc.num_sub_units
"""

df = pd.read_sql_query(query, con)

# Compute efficiency metrics per run
df['param_efficiency'] = (1 / df['test_loss']) / df['unit_parameters']
df['wall_clock_efficiency'] = (1 / df['test_loss']) / df['duration']
df['gpu_forward_efficiency'] = (1 / df['test_loss']) / df['time_forward_gpu']
df['gpu_backward_efficiency'] = (1 / df['test_loss']) / df['time_backward_gpu']
df['cpu_forward_efficiency'] = (1 / df['test_loss']) / df['time_forward_cpu']
df['cpu_backward_efficiency'] = (1 / df['test_loss']) / df['time_backward_cpu']

# Aggregate comparable configurations
group_cols = ["unit_type", "unit_params", "data_split", "data_batch_size", "global_seed",
              "optimizer", "criterion", "learning_rate", "dropout"]

# Ensure timing values are consistent within each group
constant_cols = [
    "time_forward_gpu", "time_backward_gpu", "time_forward_cpu", "time_backward_cpu"
]
nunique_per_group = df.groupby(group_cols)[constant_cols].nunique(dropna=False)
inconsistent_mask = (nunique_per_group > 1).any(axis=1)
if inconsistent_mask.any():
    inconsistent_groups = nunique_per_group[inconsistent_mask]
    raise ValueError(
        "Inconsistent compute/timing values detected within grouped configurations. "
        f"Found {len(inconsistent_groups)} inconsistent group(s)."
    )

agg_dict = {col: ["mean", "std"] for col in ["best_loss", "test_loss", "test_accuracy",
                                               "test_f1_score", "test_auroc", "duration",
                                               "epoch_stopped", "unit_sub_units", "unit_parameters",
                                               "param_efficiency", "wall_clock_efficiency",
                                               "gpu_forward_efficiency", "gpu_backward_efficiency",
                                               "cpu_forward_efficiency", "cpu_backward_efficiency"]}
agg_dict.update({
    "time_forward_gpu": "first",
    "time_backward_gpu": "first",
    "time_forward_cpu": "first",
    "time_backward_cpu": "first"
})

summary_stats = df.groupby(group_cols).agg(agg_dict).round(6)

# Flatten the aggregated column names
summary_stats.columns = [
    col[0] if (isinstance(col, tuple) and col[1] == "first")
    else (f"{col[0]}_{col[1]}" if isinstance(col, tuple) else col)
    for col in summary_stats.columns
]
summary_stats = summary_stats.reset_index()

summary_stats.to_sql("run_summary_stats", con, if_exists="replace", index=False)
con.commit()

# Select the top configuration for each unit type
df = pd.read_sql_query("SELECT * FROM run_summary_stats", con)
best_configs = df.loc[df.groupby(["unit_type"])["test_accuracy_mean"].idxmax()].copy()

units_to_exclude = ["SigmoidUnit", "UniversalSelfGatingUnit", "SelfAttentionUnit"]
rename_mapping = {
    "UniversalMaxUnit": "Maxout",
    "SoftmaxUnit": "MoE",
    "BilinearRowGatingUnit": "BRGU",
    "BilinearMixtureGatingUnit": "BMGU",
    "BilinearSelfGatingUnit": "BDGU",
}

filtered = best_configs[~best_configs["unit_type"].isin(units_to_exclude)].copy()
filtered["unit_type"] = filtered["unit_type"].replace(rename_mapping)

# Build the ESANN-style summary table
summary = filtered.set_index("unit_type")

def add_delta_vs_moe(df, value_col, out_col):
    if "MoE" not in df.index:
        df[out_col] = np.nan
        return
    moe_val = df.loc["MoE", value_col]
    df[out_col] = (df[value_col] / moe_val - 1.0) * 100.0
    df.loc["MoE", out_col] = 0.0

add_delta_vs_moe(summary, "test_loss_mean", "delta_test_loss_vs_moe")
add_delta_vs_moe(summary, "test_accuracy_mean", "delta_test_accuracy_vs_moe")
add_delta_vs_moe(summary, "test_auroc_mean", "delta_test_auroc_vs_moe")
add_delta_vs_moe(summary, "cpu_forward_efficiency_mean", "delta_eff_cpu_fwd_vs_moe")
add_delta_vs_moe(summary, "cpu_backward_efficiency_mean", "delta_eff_cpu_bwd_vs_moe")
add_delta_vs_moe(summary, "gpu_forward_efficiency_mean", "delta_eff_gpu_fwd_vs_moe")
add_delta_vs_moe(summary, "gpu_backward_efficiency_mean", "delta_eff_gpu_bwd_vs_moe")
col_precision = {"test_accuracy": 2, "test_auroc": 2}
for col in ["test_loss", "test_accuracy", "test_f1_score", "test_auroc"]:
    if f"{col}_mean" in summary.columns:
        p = col_precision.get(col, 4)
        summary[f"{col}_mean_std"] = summary.apply(
            lambda row, c=col, prec=p: (
                f"{row[f'{c}_mean']:.{prec}f}±{row[f'{c}_std']:.{prec}f}"
                if pd.notnull(row[f'{c}_mean']) and pd.notnull(row.get(f'{c}_std', np.nan))
                else np.nan
            ),
            axis=1
        )

TINY_OUT_FILE = RESULT_DIR / "generated_tiny_imagenet_table.csv"
ORDER = ["BDGU", "BMGU", "MoE", "Maxout", "BRGU"]

headers = [
    "unit_type",
    "test_loss_mean", "test_loss_std", "test_loss_mean_std", "delta_test_loss_vs_moe",
    "test_accuracy_mean", "test_accuracy_std", "test_accuracy_mean_std", "delta_test_accuracy_vs_moe",
    "test_auroc_mean", "test_auroc_std", "test_auroc_mean_std", "delta_test_auroc_vs_moe",
    "cpu_forward_efficiency_mean", "delta_eff_cpu_fwd_vs_moe",
    "cpu_backward_efficiency_mean", "delta_eff_cpu_bwd_vs_moe",
    "gpu_forward_efficiency_mean", "delta_eff_gpu_fwd_vs_moe",
    "gpu_backward_efficiency_mean", "delta_eff_gpu_bwd_vs_moe",
]
row_cols = headers[1:]

rows = []
for unit in ORDER:
    vals = summary.loc[unit, row_cols].tolist() if unit in summary.index else [np.nan] * len(row_cols)
    rows.append([unit] + vals)

accuracy_auroc_formats = {c: ".2f" for c in [
    "test_accuracy_mean", "test_accuracy_std", "delta_test_accuracy_vs_moe",
]}

auroc_formats = {c: ".4f" for c in [
    "test_auroc_mean", "test_auroc_std", "delta_test_auroc_vs_moe",
]}

write_csv(TINY_OUT_FILE, headers, rows, col_formats={**accuracy_auroc_formats, **auroc_formats})
print("Saved Tiny-ImageNet aggregated table!")

#%%
# Save per-unit parameter summaries to CSV
MAX_PARAMS = 400_000

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
    )
)

summary["avg_rel_to_max"] = (summary["avg_unit_parameters"] / MAX_PARAMS) * 100.0

summary = summary[[
    "unit_type",
    "min_unit_parameters",
    "max_unit_parameters",
    "avg_unit_parameters",
    "avg_rel_to_max",
]].sort_values("unit_type")

PARAM_SUMMARY_OUT_FILE = RESULT_DIR / "generated_tiny_imagenet_parameter_summary_table.csv"
summary.to_csv(PARAM_SUMMARY_OUT_FILE, index=False, float_format="%.6f")

print(f"Saved parameter summary CSV: {PARAM_SUMMARY_OUT_FILE}")
print(summary)

# Compute parameter variance across all runs
overall_mean = summary["max_unit_parameters"].mean()
overall_std = summary["max_unit_parameters"].std()
print(f"Overall mean of unit parameters across all runs: {overall_mean}")
print(f"Overall standard deviation of unit parameters across all runs: {overall_std}")
con.close()