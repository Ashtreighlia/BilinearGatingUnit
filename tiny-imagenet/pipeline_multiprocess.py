import pickle
import torch
import numpy as np
import random
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import os
import tqdm
import gc
import sys
import torch.nn as nn

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import dataset
from pipeline_train import pipeline

import units
from modules import set_global_seed, Normalizer


def initialize_unit(unit_type, input_dim, out_features, num_sub_units, dropout=0, seed=None):
    unit_name, scaling_mode = units.parse_unit_type(unit_type)

    if hasattr(units, unit_name):
        unit_class = getattr(units, unit_name)
        return unit_class(
            in_features=input_dim,
            out_features=out_features,
            sub_units=num_sub_units,
            dropout=dropout,
            seed=seed,
            scaling_mode=scaling_mode,
        )
    else:
        raise ValueError(f"Unknown unit: {unit_name}")


def optimizer_factory(encoder, unit, optimizer_type, lr):
    if optimizer_type == "adam":
        return torch.optim.Adam(list(encoder.parameters()) + list(unit.parameters()), lr=lr)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_type}")


def criterion_factory(criterion_type):
    if criterion_type == "mse":
        return torch.nn.MSELoss()
    elif criterion_type == "mae":
        return torch.nn.L1Loss()
    elif criterion_type == "ce":
        return torch.nn.CrossEntropyLoss()
    else:
        raise ValueError(f"Unknown criterion: {criterion_type}")

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.act1 = nn.LeakyReLU()
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)

        return x

class SimpleEncoder(torch.nn.Module):
    def __init__(self):
        super(SimpleEncoder, self).__init__()

        self.encoder = nn.Sequential(
            ConvBlock(3, 16),                                            # 64x64 -> 64x64
            nn.MaxPool2d(kernel_size=2),                                # 64x64 -> 32x32

            ConvBlock(16, 32),                                           # 32x32 -> 32x32
            nn.MaxPool2d(kernel_size=2),                                # 32x32 -> 16x16

            ConvBlock(32, 64),                                          # 16x16 -> 16x16
            nn.MaxPool2d(kernel_size=2),                                # 16x16 -> 8x8

            ConvBlock(64, 128),                                         # 8x8 -> 8x8
            nn.AdaptiveAvgPool2d((1, 1)),                               # 8x8 -> 1x1
            nn.Flatten(),                                               # 128 x 1 x 1 -> 128

            nn.LayerNorm(128),
        )

    def forward(self, x):
        return self.encoder(x)
    

class ClassificationHead(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.num_sub_units = units.calculate_sub_units(
            unit_type=cfg["unit_type"],
            input_dim=128,
            output_dim=128,
            num_params=cfg["unit_params"]//2,
        )

        self.unit_1 = initialize_unit(
            unit_type=cfg["unit_type"],
            input_dim=128,
            out_features=128,
            num_sub_units=self.num_sub_units,
            dropout=cfg["dropout"],
            seed=cfg["seed"],
        )

        self.unit_2 = initialize_unit(
            unit_type=cfg["unit_type"],
            input_dim=128,
            out_features=128,
            num_sub_units=self.num_sub_units,
            dropout=cfg["dropout"],
            seed=cfg["seed"] + 1,
        )

        self.final_layer = nn.Linear(128, 200)
    
    def get_parameter_count(self):
        params = units.get_parameter_count(self.unit_1) + units.get_parameter_count(self.unit_2)
        return params
    
    def get_sub_units(self):
        return self.num_sub_units

    def forward(self, x):
        skip = x
        y = self.unit_1(x)
        y = y + skip  # Skip connection
        skip = y
        y = self.unit_2(y)
        y = y + skip  # Skip connection
        y = self.final_layer(y)
        return y


def training_worker(cfg, device):
    process_id = cfg["id"]

    torch.multiprocessing.set_sharing_strategy('file_system')

    # Set global seed for reproducibility
    set_global_seed(cfg["global_seed"])
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    encoder = SimpleEncoder().to(device)

    classification_head = ClassificationHead(cfg).to(device)

    unit_parameter_count = classification_head.get_parameter_count()

    optimizer = optimizer_factory(encoder, classification_head, cfg["optimizer"], cfg["learning_rate"])
    criterion = criterion_factory(cfg["criterion"])

    data = dataset.TinyImageNetDataset(
        val_split=cfg["data_split"],
        seed=cfg["global_seed"],
    )

    train_loader, val_loader, test_loader = data.get_dataloaders(
        batch_size=cfg["data_batch_size"]
    )

    result = pipeline(
        device,
        encoder,
        classification_head,
        train_loader,
        val_loader,
        test_loader,
        optimizer,
        criterion,
        max_epochs=cfg["max_epochs"],
    )

    (
        best_encoder,
        best_model,
        best_loss,
        test_loss,
        test_accuracy,
        test_f1,
        test_auroc,
        log,
        duration,
        epoch_stopped,
    ) = result

    # Serialize CPU state dicts before returning results to the parent process
    best_encoder = best_encoder.cpu()
    best_model = best_model.cpu()

    encoder_blob = pickle.dumps(best_encoder.state_dict())
    model_blob = pickle.dumps(best_model.state_dict())

    return_dict = {
        "process_id": process_id,
        "best_encoder_blob": encoder_blob,
        "best_model_blob": model_blob,
        "best_loss": best_loss,
        "test_loss": test_loss,
        "test_accuracy": test_accuracy,
        "test_f1": test_f1,
        "test_auroc": test_auroc,
        "duration": duration,
        "epoch_stopped": epoch_stopped,
        "log": log,
        "unit_sub_units": classification_head.get_sub_units(),
        "unit_parameters": unit_parameter_count,
    }

    # Release the data loaders, classifier, and checkpoints before returning
    del train_loader, val_loader, test_loader, data
    del classification_head, optimizer, best_model, best_encoder

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
        """Save training result to database"""
        try:
            encoder_blob = result["best_encoder_blob"]
            model_blob = result["best_model_blob"]

            duration = result.get("duration", None)
            epoch_stopped = result.get("epoch_stopped", None)
            log_blob = pickle.dumps(result["log"])

            unit_sub_units = result.get("unit_sub_units", None)
            unit_parameters = result.get("unit_parameters", None)

            db_cur.execute(
                """
                INSERT INTO run_results (run_id, best_loss, test_loss, test_accuracy, test_f1_score, test_auroc, duration, epoch_stopped, unit_sub_units, unit_parameters)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    result["process_id"],
                    result["best_loss"],
                    result["test_loss"],
                    result["test_accuracy"],
                    result["test_f1"],
                    result["test_auroc"],
                    duration,
                    epoch_stopped,
                    unit_sub_units,
                    unit_parameters,
                ),
            )

            db_cur.execute(
                """
                INSERT INTO run_blobs (run_id, best_encoder, best_model, log)
                VALUES (?, ?, ?, ?)
            """,
                (result["process_id"], encoder_blob, model_blob, log_blob),
            )

            db_cur.connection.commit()

            # Release serialized buffers after committing the result
            if result is not None:
                del result
            del encoder_blob, model_blob, log_blob
            gc.collect()

        except Exception as e:
            print(f"Failed to save process {result['process_id']}: {e}")

    # Track per-GPU load to enforce the concurrency limit
    gpu_job_counts = {gpu_id: 0 for gpu_id in range(num_gpus)}
    max_concurrent = num_gpus * concurrent_gpu_jobs

    with ProcessPoolExecutor(max_workers=max_concurrent) as executor:
        active_futures = {}  # future -> (cfg, gpu_id)
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
                        for future in as_completed(active_futures.keys(), timeout=1):
                            done_futures.append(future)
                            break  # Handle one completion at a time to rebalance GPU assignments
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
