import math
import torch
from torch.utils.data import Dataset


class SyntheticData(Dataset):
    """Generate a synthetic function over a regular coordinate grid."""

    def __init__(self, data_type, dimensions, resolution, seed=None, train_val_split=None):
        self.dimensions = dimensions
        # Use the same number of points per axis; the resulting total may be below resolution
        per_dim_resolution = int(resolution ** (1 / dimensions))

        # Create a regular grid spanning [-pi, pi] in each dimension
        coords = [torch.linspace(-math.pi, math.pi, per_dim_resolution) for _ in range(self.dimensions)]
        meshgrids = torch.meshgrid(*coords, indexing='ij')
        # Store each grid point as one row of coordinates
        self.coords = torch.stack([grid.flatten() for grid in meshgrids], dim=1)

        # Evaluate the selected function at every grid point
        if data_type == "x2":
            self.z = self.x2(self.coords)
        elif data_type == "exp":
            self.z = self.exp(self.coords)
        elif data_type == "log":
            self.z = self.log(self.coords)
        elif data_type == "x3":
            self.z = self.x3(self.coords)
        elif data_type == "poly":
            self.z = self.poly(self.coords)
        elif data_type == "sin":
            self.z = self.sin(self.coords)
        elif data_type == "sincos":
            self.z = self.sincos(self.coords)
        elif data_type == "sin4cos":
            self.z = self.sin4cos(self.coords)
        elif data_type == "wave":
            self.z = self.wave(self.coords)
        elif data_type == "abs":
            self.z = self.abs(self.coords)
        else:
            raise ValueError(f"Unknown data_type: {data_type}")

        # Shuffle and split the grid when train/validation fractions are provided
        if train_val_split is not None:
            if round(sum(train_val_split.values()), 10) != 1:
                raise ValueError("Train/Val split must sum to 1.")
            
            if seed is not None:
                generator = torch.Generator().manual_seed(seed)
                perm = torch.randperm(len(self.coords), generator=generator)
            else:
                print("No seed provided for data shuffling.")
                perm = torch.randperm(len(self.coords))

            self.coords_shuffle = self.coords[perm]
            self.z_shuffle = self.z[perm]

            # Any samples left after these rounded sizes are assigned to the test set
            train_size = int(train_val_split["train"] * len(self.coords))
            val_size = int(train_val_split["val"] * len(self.coords))

            self.train_coords = self.coords_shuffle[:train_size]
            self.train_z = self.z_shuffle[:train_size]
            self.val_coords = self.coords_shuffle[train_size:train_size + val_size]
            self.val_z = self.z_shuffle[train_size:train_size + val_size]
            self.test_coords = self.coords_shuffle[train_size + val_size:]
            self.test_z = self.z_shuffle[train_size + val_size:]

    def x2(self, coords):
        # Sum of squares across all dimensions
        return torch.sum(coords**2, dim=1)

    def exp(self, coords):
        # Sum of exponentials across all dimensions
        return torch.sum(torch.exp(coords), dim=1)

    def log(self, coords):
        # Sum of logs across all dimensions
        return torch.sum(torch.log(torch.abs(coords) + 1e-5), dim=1)

    def x3(self, coords):
        # Sum of cubes across all dimensions
        return torch.sum(coords**3, dim=1)
    
    def poly(self, coords):
        # Sum of polynomial (x^4 - x^2 + x) across all dimensions
        return torch.sum(coords**4 - coords**2 + coords, dim=1)
    
    def sin(self, coords):
        # Sum of sines across all dimensions
        return torch.sum(torch.sin(coords), dim=1)

    def sincos(self, coords):
        # Sum of sin(cos(x)) across all dimensions
        return torch.sum(torch.sin(torch.cos(coords)), dim=1)
    
    def sin4cos(self, coords):
        # Sum of sin(4 * cos(x)) across all dimensions
        return torch.sum(torch.sin(4 * torch.cos(coords)), dim=1)
    
    def wave(self, coords):
        result = torch.zeros(coords.shape[0], device=coords.device)
        for i in range(self.dimensions):
            base = torch.sin(coords[:, i]) * (i+1)

            if i < self.dimensions - 1:
                # Couple adjacent dimensions to vary each component's contribution
                coupling = torch.cos(coords[:, i] * coords[:, i+1])
                base = base * (1 + 0.5*coupling)
        
            # Modulate each component's amplitude before adding it to the result
            amplitude = 1 + 0.25 * torch.sin(2*coords[:, i])
            result += amplitude * base

        return result

    def abs(self, coords):
        # Sum of absolute values across all dimensions
        return torch.sum(torch.abs(coords), dim=1)

    def __len__(self):
        return len(self.coords)

    def __getitem__(self, idx):
        return self.coords[idx], self.z[idx]

    def get_all(self, split=None):
        # Return the requested split, or the full unsplit dataset by default
        if split == "train":
            return self.train_coords, self.train_z
        elif split == "val":
            return self.val_coords, self.val_z
        elif split == "test":
            return self.test_coords, self.test_z
        else:
            return self.coords, self.z