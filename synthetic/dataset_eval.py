#%%
import math
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from dataset import SyntheticData

#%%
# Dataset types to visualize
data_types = [
    "x2",
    "exp",
    "log",
    "x3",
    "poly",
    "sin",
    "sincos",
    "sin4cos",
    "wave",
    "abs",
]
dimensions_to_test = [1, 2, 3]
resolution = 10000  # Total number of points

# Create output directory
save_dir = Path(__file__).resolve().parent.parent / "plots" / "synthetic_data"
save_dir.mkdir(parents=True, exist_ok=True)

for dim in dimensions_to_test:
    sub_folder = save_dir / f"dim{dim}"
    sub_folder.mkdir(parents=True, exist_ok=True)
    for dtype in data_types:
        fig = plt.figure(figsize=(5, 5))

        # Generate samples for this function and dimension
        data = SyntheticData(data_type=dtype, dimensions=dim, resolution=resolution)
        coords, z = data.get_all()

        if dim == 3:
            ax = fig.add_subplot(1, 1, 1, projection="3d")
        else:
            ax = fig.add_subplot(1, 1, 1)

        if dim == 1:
            # Sort points for a continuous curve
            sorted_idx = torch.argsort(coords.flatten())
            ax.plot(coords[sorted_idx].cpu().numpy(), z[sorted_idx].cpu().numpy(), linewidth=4)
            ax.set_xlabel("x", fontsize=24)
            ax.set_ylabel("y", fontsize=24)
            ax.tick_params(axis='both', which='major', labelsize=18)

        elif dim == 2:
            # Restore sampled values to a 2D grid
            side_length = int(len(z) ** 0.5)
            z_grid = z.reshape(side_length, side_length).cpu().numpy()

            # Transpose so x1 runs horizontally
            extent = [-math.pi, math.pi, -math.pi, math.pi]
            im = ax.imshow(z_grid.T, extent=extent, origin="lower", cmap="viridis")
            ax.set_xlabel("x1", fontsize=24)
            ax.set_ylabel("x2", fontsize=24)
            ax.tick_params(axis='both', which='major', labelsize=18)
            cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.ax.tick_params(labelsize=18)

        elif dim == 3:
            # Color each 3D input point by its output value
            ax.scatter(
                coords[:, 0].cpu().numpy(),
                coords[:, 1].cpu().numpy(),
                coords[:, 2].cpu().numpy(),
                c=z.cpu().numpy(),
                cmap="viridis",
                s=100,
                alpha=0.05,
            )
            ax.set_xlabel("x1", fontsize=24)
            ax.set_ylabel("x2", fontsize=24)
            ax.set_zlabel("x3", fontsize=24)
            ax.tick_params(axis='x', which='major', labelsize=18)
            ax.tick_params(axis='y', which='major', labelsize=18)
            ax.tick_params(axis='z', which='major', labelsize=18)

        fig.tight_layout()

        filename = f"data_{dtype}.png"
        filepath = sub_folder / filename
        # Keep 3D labels within the saved figure
        plt.savefig(filepath, bbox_inches='tight')

        plt.show()
        plt.close(fig)