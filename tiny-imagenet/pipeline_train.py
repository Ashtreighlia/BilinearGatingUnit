import copy
import time
import sys
import os

import torch
from torcheval.metrics import MulticlassF1Score, MulticlassAUROC

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from modules import EarlyStopping

def pipeline(
    device,
    encoder,
    unit,
    train_loader,
    val_loader,
    test_loader,
    optimizer,
    criterion,
    max_epochs=None,
):
    log = {}
    log["train_loss"] = {}
    log["val_loss"] = {}
    log["train_accuracy"] = {}
    log["val_accuracy"] = {}

    encoder.to(device)
    unit.to(device)

    best_loss = float("inf")
    best_model = None
    best_encoder = None

    start_time = time.time()
    current_epoch = 0
    early_stopping = EarlyStopping(patience=10, delta=0, decay=0.01)

    # Tiny ImageNet has 200 classes
    num_classes = 200

    while True:
        # Training phase
        encoder.train()
        unit.train()
        total_train_loss = 0
        correct_train = 0
        total_train = 0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            # Normalize image values to [-1, 1]
            images = (images - 0.5) / 0.5

            images = encoder(images)
            labels = labels.squeeze().long()  # Convert class labels to integers
            
            optimizer.zero_grad()
            outputs = unit(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

            # Release batch tensors before the next iteration
            del outputs, loss, predicted, images, labels

        optimizer.zero_grad()

        train_loss = total_train_loss / len(train_loader)
        train_accuracy = 100 * correct_train / total_train
        
        log["train_loss"][str(current_epoch)] = train_loss
        log["train_accuracy"][str(current_epoch)] = train_accuracy

        # Validation phase
        encoder.eval()
        unit.eval()
        total_val_loss = 0
        correct_val = 0
        total_val = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)

                # Normalize image values to [-1, 1]
                images = (images - 0.5) / 0.5

                images = encoder(images)
                labels = labels.squeeze().long()  # Convert class labels to integers
                
                outputs = unit(images)
                loss = criterion(outputs, labels)
                
                total_val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()

                # Release batch tensors before the next iteration
                del outputs, loss, predicted, images, labels

        val_loss = total_val_loss / len(val_loader)
        val_accuracy = 100 * correct_val / total_val
        
        log["val_loss"][str(current_epoch)] = val_loss
        log["val_accuracy"][str(current_epoch)] = val_accuracy

        # Retain the checkpoint with the lowest validation loss
        if val_loss < best_loss:
            best_loss = val_loss
            del best_model
            best_model = copy.deepcopy(unit)
            del best_encoder
            best_encoder = copy.deepcopy(encoder)

        if early_stopping(val_loss) or (
            max_epochs is not None and current_epoch >= max_epochs
        ):
            duration = time.time() - start_time
            break

        current_epoch += 1

    # Test phase
    best_model.eval()
    best_encoder.eval()
    total_test_loss = 0
    correct_test = 0
    total_test = 0
    
    test_f1_metric = MulticlassF1Score(num_classes=num_classes, device=device)
    test_auroc_metric = MulticlassAUROC(num_classes=num_classes, device=device)
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)

            # Normalize image values to [-1, 1]
            images = (images - 0.5) / 0.5

            images = best_encoder(images)
            labels = labels.squeeze().long()  # Convert class labels to integers
            
            outputs = best_model(images)
            loss = criterion(outputs, labels)
            
            total_test_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_test += labels.size(0)
            correct_test += (predicted == labels).sum().item()
            
            probabilities = torch.softmax(outputs, dim=1)
            test_f1_metric.update(predicted, labels)
            test_auroc_metric.update(probabilities, labels)

    test_loss = total_test_loss / len(test_loader)
    test_accuracy = 100 * correct_test / total_test
    test_f1 = test_f1_metric.compute().item()
    test_auroc = test_auroc_metric.compute().item()

    del test_f1_metric, test_auroc_metric
    
    log["test_loss"] = test_loss
    log["test_accuracy"] = test_accuracy
    log["test_f1"] = test_f1
    log["test_auroc"] = test_auroc

    return (
        best_encoder,
        best_model,
        best_loss,
        test_loss,
        test_accuracy,
        test_f1,
        test_auroc,
        log,
        duration,
        current_epoch,
    )