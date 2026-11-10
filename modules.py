import math
import random

import torch
import numpy as np

def set_global_seed(seed):
    """Seed Python, NumPy, and PyTorch and enable deterministic cuDNN behavior"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class EarlyStopping:
    """Stop when validation loss no longer improves by the required margin

    The margin scales with the loss magnitude when decay is set
    """

    def __init__(self, patience=5, delta=0, decay=None):
        self.patience = patience
        self.base_delta = delta
        self.decay = decay
        self.delta = delta
        self.best_score = None
        self.counter = 0
        self.early_stop = False
        self.val_loss_min = float('inf')

    def __call__(self, val_loss):
        score = -val_loss
        
        if self.decay is not None and not math.isnan(val_loss):
            # Scale the required improvement with the current loss magnitude
            self.delta = self.base_delta + (abs(val_loss) * self.decay)
        
        if self.best_score is None:
            self.best_score = score
            self.val_loss_min = val_loss
        elif math.isnan(score):
            self.early_stop = True
        else:
            if self.delta == 0:
                not_improved = score < self.best_score
            else:
                not_improved = score < self.best_score + self.delta

            if not_improved:
                self.counter += 1
                if self.counter >= self.patience:
                    self.early_stop = True
            else:
                self.best_score = score
                self.val_loss_min = val_loss
                self.counter = 0
                self.early_stop = False

        return self.early_stop
    
class Normalizer:
    """Normalize data using stored feature-wise mean and standard deviation"""

    def __init__(self, data, device='cpu'):
        self.mean = data.mean(dim=0, keepdim=True)
        self.std = data.std(dim=0, keepdim=True) + 1e-8
        self.mean = self.mean.to(device)
        self.std = self.std.to(device)

    def normalize(self, data):
        """Apply the stored mean and standard deviation"""
        return (data - self.mean) / self.std

    def denormalize(self, data):
        """Restore data to its original scale"""
        return data * self.std + self.mean