import ast
import random
import os
import sys
import gc

import pickle
import torch
import numpy as np
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import dataset
from pipeline_train import pipeline

import units
from modules import set_global_seed, Normalizer

def initialize_unit(unit_type, input_dim, num_sub_units, dropout=0, seed=None):
    unit_name, scaling_mode = units.parse_unit_type(unit_type)
    if hasattr(units, unit_name):
        unit_class = getattr(units, unit_name)
        return unit_class(
            in_features=input_dim, out_features=1, sub_units=num_sub_units, dropout=dropout, seed=seed, scaling_mode=scaling_mode
        )
    else:
        raise ValueError(f"Unknown unit: {unit_name}")


def optimizer_factory(unit, optimizer_type, lr):
    if optimizer_type == "adam":
        return torch.optim.Adam(unit.parameters(), lr=lr)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_type}")


def criterion_factory(criterion_type):
    if criterion_type == "mse":
        return torch.nn.MSELoss()
    elif criterion_type == "mae":
        return torch.nn.L1Loss()
    else:
        raise ValueError(f"Unknown criterion: {criterion_type}")


def training_worker(cfg, device):
    process_id = cfg["id"]

    # Set global seed for reproducibility
    set_global_seed(cfg["global_seed"])

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Configurations loaded from SQLite store data_splits as text
    if isinstance(cfg["data_splits"], str):
        data_splits = ast.literal_eval(cfg["data_splits"])
    else:
        data_splits = cfg["data_splits"]

    num_sub_units = units.calculate_sub_units(
        unit_type=cfg["unit_type"],
        input_dim=cfg["data_dimension"],
        output_dim=1,
        num_params=cfg["unit_params"],
    )

    unit = initialize_unit(
        unit_type=cfg["unit_type"],
        input_dim=cfg["data_dimension"],
        num_sub_units=num_sub_units,
        dropout=cfg["dropout"],
        seed=cfg["seed"],
    )

    unit_parameter_count = units.get_parameter_count(unit)

    optimizer = optimizer_factory(unit, cfg["optimizer"], cfg["learning_rate"])
    criterion = criterion_factory(cfg["criterion"])

    data = dataset.SyntheticData(
        data_type=cfg["data_type"],
        dimensions=cfg["data_dimension"],
        resolution=cfg["data_resolution"],
        seed=cfg["global_seed"],
        train_val_split=data_splits,
    )

    # Overfitting mode: train and evaluate on the entire dataset
    if cfg["data_mode"] == "overfitting":
        xy_train, z_train = data.get_all()

        normalizer_xy_train = Normalizer(xy_train)
        normalizer_z_train = Normalizer(z_train)
        xy_train = normalizer_xy_train.normalize(xy_train)
        z_train = normalizer_z_train.normalize(z_train)

        result = pipeline(
            device,
            unit,
            xy_train,
            z_train,
            optimizer,
            criterion,
            max_epochs=cfg["max_epochs"],
        )

        best_model, best_loss, train_data, log, duration, epoch_stopped = result

        # Keep returned values on CPU for process-pool transfer
        best_model = best_model.cpu()

        if isinstance(train_data, tuple):
            train_data = tuple(
                t.detach().cpu() if torch.is_tensor(t) else t for t in train_data
            )
        else:
            train_data = (
                train_data.detach().cpu() if torch.is_tensor(train_data) else train_data
            )

        return_dict = {
            "process_id": process_id,
            "best_model": best_model,
            "best_loss": best_loss,
            "train_data": train_data,
            "duration": duration,
            "epoch_stopped": epoch_stopped,
            "log": log,
            "unit_sub_units": num_sub_units,
            "unit_parameters": unit_parameter_count,
        }

        # Release training tensors and GPU state
        del xy_train, z_train, unit, optimizer
        del normalizer_xy_train, normalizer_z_train
        torch.cuda.empty_cache()
        gc.collect()

        return return_dict

    # Generalization mode: train on training data and evaluate on held-out splits
    elif cfg["data_mode"] == "generalization":
        xy_train, z_train = data.get_all("train")
        xy_val, z_val = data.get_all("val")
        xy_test, z_test = data.get_all("test")

        # Fit on training data only, then reuse the statistics for validation and test
        normalizer_xy = Normalizer(xy_train)
        normalizer_z = Normalizer(z_train)

        xy_train = normalizer_xy.normalize(xy_train).to(device)
        z_train = normalizer_z.normalize(z_train).to(device)
        xy_val = normalizer_xy.normalize(xy_val).to(device)
        z_val = normalizer_z.normalize(z_val).to(device)
        xy_test = normalizer_xy.normalize(xy_test).to(device)
        z_test = normalizer_z.normalize(z_test).to(device)

        result = pipeline(
            device,
            unit,
            xy_train,
            z_train,
            optimizer,
            criterion,
            xy_val=xy_val,
            z_val=z_val,
            xy_test=xy_test,
            z_test=z_test,
            max_epochs=cfg["max_epochs"],
        )

        (
            best_model,
            best_loss,
            train_data,
            val_data,
            test_data,
            log,
            duration,
            epoch_stopped,
        ) = result

        # Keep returned values on CPU for process-pool transfer
        best_model = best_model.cpu()

        if isinstance(train_data, tuple):
            train_data = tuple(
                t.detach().cpu() if torch.is_tensor(t) else t for t in train_data
            )
        else:
            train_data = (
                train_data.detach().cpu() if torch.is_tensor(train_data) else train_data
            )

        if isinstance(val_data, tuple):
            val_data = tuple(
                t.detach().cpu() if torch.is_tensor(t) else t for t in val_data
            )
        else:
            val_data = (
                val_data.detach().cpu() if torch.is_tensor(val_data) else val_data
            )

        if isinstance(test_data, tuple):
            test_data = tuple(
                t.detach().cpu() if torch.is_tensor(t) else t for t in test_data
            )
        else:
            test_data = (
                test_data.detach().cpu() if torch.is_tensor(test_data) else test_data
            )

        return_dict = {
            "process_id": process_id,
            "best_model": best_model,
            "best_loss": best_loss,
            "train_data": train_data,
            "val_data": val_data,
            "test_data": test_data,
            "duration": duration,
            "epoch_stopped": epoch_stopped,
            "log": log,
            "unit_sub_units": num_sub_units,
            "unit_parameters": unit_parameter_count,
        }

        # Release training tensors and GPU state before the next run
        del xy_train, z_train, xy_val, z_val, xy_test, z_test
        del unit, optimizer
        del normalizer_xy, normalizer_z
        torch.cuda.empty_cache()
        gc.collect()

        return return_dict


def worker_with_gpu(args):
    cfg, gpu_id = args
    device = f"cuda:{gpu_id}"
    return training_worker(cfg, device)


def parallel_training(db_cur, cfgs, concurrent_gpu_jobs=1):
    num_gpus = torch.cuda.device_count()
    print(f"Detected {num_gpus} GPUs")

    def save_result_to_db(result):
        try:
            best_model = result["best_model"]
            model_blob = pickle.dumps(best_model.state_dict())

            duration = result.get("duration", None)
            epoch_stopped = result.get("epoch_stopped", None)
            log_blob = pickle.dumps(result["log"])
            train_data_blob = pickle.dumps(result["train_data"])

            unit_sub_units = result.get("unit_sub_units", None)
            unit_parameters = result.get("unit_parameters", None)

            if "val_data" in result and "test_data" in result:
                # Generalization runs also store validation and test data
                val_data_blob = pickle.dumps(result["val_data"])
                test_data_blob = pickle.dumps(result["test_data"])

                db_cur.execute(
                    """
                    INSERT INTO run_results (run_id, best_loss, duration, epoch_stopped, unit_sub_units, unit_parameters)
                    VALUES (?, ?, ?, ?, ?, ?)
                """,
                    (
                        result["process_id"],
                        result["best_loss"],
                        duration,
                        epoch_stopped,
                        unit_sub_units,
                        unit_parameters,
                    ),
                )

                db_cur.execute(
                    """
                    INSERT INTO run_blobs (run_id, best_model, log, train_data, val_data, test_data)
                    VALUES (?, ?, ?, ?, ?, ?)
                """,
                    (
                        result["process_id"],
                        model_blob,
                        log_blob,
                        train_data_blob,
                        val_data_blob,
                        test_data_blob,
                    ),
                )
            else:
                # Overfitting runs have no validation or test data
                db_cur.execute(
                    """
                    INSERT INTO run_results (run_id, best_loss, duration, epoch_stopped, unit_sub_units, unit_parameters)
                    VALUES (?, ?, ?, ?, ?, ?)
                """,
                    (
                        result["process_id"],
                        result["best_loss"],
                        duration,
                        epoch_stopped,
                        unit_sub_units,
                        unit_parameters,
                    ),
                )

                db_cur.execute(
                    """
                    INSERT INTO run_blobs (run_id, best_model, log, train_data, val_data, test_data)
                    VALUES (?, ?, ?, ?, ?, ?)
                """,
                    (
                        result["process_id"],
                        model_blob,
                        log_blob,
                        train_data_blob,
                        None,
                        None,
                    ),
                )

            db_cur.connection.commit()

            # Release serialized buffers after committing the result
            if result is not None:
                del result
            del model_blob, log_blob, train_data_blob
            if "val_data_blob" in locals():
                del val_data_blob
            if "test_data_blob" in locals():
                del test_data_blob
            gc.collect()

        except Exception as e:
            print(f"Failed to save process {result['process_id']}: {e}")

    # Track per-GPU load to enforce the concurrency limit
    gpu_job_counts = {gpu_id: 0 for gpu_id in range(num_gpus)}
    max_concurrent = num_gpus * concurrent_gpu_jobs

    with ProcessPoolExecutor(max_workers=max_concurrent) as executor:
        active_futures = {}
        cfg_queue = list(cfgs)
        completed = 0
        total = len(cfgs)

        with tqdm.tqdm(total=total, desc="Training Progress") as pbar:
            while completed < total:
                # Fill available slots, preferring the least-loaded GPU
                while cfg_queue and any(
                    count < concurrent_gpu_jobs for count in gpu_job_counts.values()
                ):
                    available_gpu = min(
                        gpu_job_counts.keys(), key=lambda x: gpu_job_counts[x]
                    )

                    if gpu_job_counts[available_gpu] < concurrent_gpu_jobs:
                        cfg = cfg_queue.pop(0)
                        job_assignment = (cfg, available_gpu)

                        future = executor.submit(worker_with_gpu, job_assignment)
                        active_futures[future] = (cfg, available_gpu)
                        gpu_job_counts[available_gpu] += 1

                if active_futures:
                    done_futures = []
                    try:
                        # Handle one completion at a time so the queue can be rebalanced
                        for future in as_completed(active_futures.keys(), timeout=1):
                            done_futures.append(future)
                            break
                    except TimeoutError:
                        continue

                    for future in done_futures:
                        cfg, gpu_id = active_futures[future]
                        result = None

                        try:
                            result = future.result()
                            save_result_to_db(result)

                            completed += 1
                            pbar.set_postfix(
                                {
                                    "Completed": f"{cfg['id']}",
                                    "GPU_Load": f"{dict(gpu_job_counts)}",
                                }
                            )
                            pbar.update(1)

                        except Exception as e:
                            # Count failed runs so progress can reach completion
                            print(f"Job {cfg['id']} on GPU {gpu_id} failed: {e}")
                            completed += 1
                            pbar.update(1)

                        finally:
                            # Release this job's GPU slot and process references
                            gpu_job_counts[gpu_id] -= 1

                            del active_futures[future]
                            if result is not None:
                                del result
                            if future is not None:
                                del future
                            gc.collect()
                            torch.cuda.empty_cache()

    print("All jobs complete!")
    torch.cuda.empty_cache()
    gc.collect()
