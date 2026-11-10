import copy
import time
import sys
import os

import torch

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from modules import EarlyStopping

def pipeline(
    device,
    unit,
    xy_train,
    z_train,
    optimizer,
    criterion,
    xy_val=None,
    z_val=None,
    xy_test=None,
    z_test=None,
    max_epochs=None,
):
    log = {}

    has_validation = xy_val is not None and z_val is not None
    has_test = xy_test is not None and z_test is not None

    if has_validation:
        log["train_loss"] = {}
        log["val_loss"] = {}

    unit.to(device)
    xy_train = xy_train.to(device)
    z_train = z_train.to(device)

    if has_validation:
        xy_val = xy_val.to(device)
        z_val = z_val.to(device)

    if has_test:
        xy_test = xy_test.to(device)
        z_test = z_test.to(device)

    best_loss = float("inf")
    best_model = None
    best_train_output = None
    best_val_output = None

    start_time = time.time()
    current_epoch = 0
    early_stopping = EarlyStopping(patience=100, delta=0, decay=0.01)

    while True:
        unit.train()
        optimizer.zero_grad()

        train_output = unit(xy_train).squeeze()
        loss = criterion(train_output, z_train)
        loss.backward()
        optimizer.step()

        train_loss_current = loss.item()

        if has_validation:
            log["train_loss"][str(current_epoch)] = train_loss_current
        else:
            log[str(current_epoch)] = {"train_loss": train_loss_current}

        # Use validation loss when available; otherwise monitor training loss
        if has_validation:
            unit.eval()
            with torch.no_grad():
                val_output = unit(xy_val).squeeze()
                val_loss = criterion(val_output, z_val).item()
                log["val_loss"][current_epoch] = val_loss

                if val_loss < best_loss:
                    best_loss = val_loss
                    best_model = copy.deepcopy(unit)
                    best_train_output = train_output
                    best_val_output = val_output

                stopping_loss = val_loss
        else:
            if train_loss_current < best_loss:
                best_loss = train_loss_current
                best_model = copy.deepcopy(unit)
                best_train_output = train_output

            stopping_loss = train_loss_current

        if early_stopping(stopping_loss) or (
            max_epochs is not None and current_epoch >= max_epochs
        ):
            duration = time.time() - start_time
            break

        current_epoch += 1

    if has_test:
        best_model.eval()
        with torch.no_grad():
            test_output = best_model(xy_test).squeeze()
            test_loss = criterion(test_output, z_test).item()
            log["test_loss"] = test_loss

    # Move returned tensors to CPU for serialization
    xy_train, z_train, best_train_output = (
        xy_train.cpu(),
        z_train.cpu(),
        best_train_output.cpu(),
    )

    if has_validation:
        xy_val, z_val, best_val_output = (
            xy_val.cpu(),
            z_val.cpu(),
            best_val_output.cpu(),
        )

    if has_test:
        xy_test, z_test, test_output = xy_test.cpu(), z_test.cpu(), test_output.cpu()

    # Return payloads for the splits that were provided
    if has_validation and has_test:
        return (
            best_model,
            best_loss,
            (xy_train, z_train, best_train_output),
            (xy_val, z_val, best_val_output),
            (xy_test, z_test, test_output),
            log,
            duration,
            current_epoch,
        )
    elif has_validation:
        return (
            best_model,
            best_loss,
            (xy_train, z_train, best_train_output),
            (xy_val, z_val, best_val_output),
            log,
            duration,
            current_epoch,
        )
    else:
        return (
            best_model,
            best_loss,
            (xy_train, z_train, best_train_output),
            log,
            duration,
            current_epoch,
        )
