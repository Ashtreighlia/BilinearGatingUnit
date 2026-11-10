import torch
from torchvision.transforms import v2 as transforms
from robust_minisets import TinyImageNet


class TinyImageNetDataset:
    """
    PyTorch Dataset wrapper for TinyImageNet.
    Wraps the TinyImageNet split and provides train/val/test DataLoaders.
    """
    def __init__(self, val_split=0.2, download=True, seed=42):
        """
        Args:
            val_split: fraction of training data to reserve for validation.
            download: whether to download the dataset on first run.
            seed: random seed for reproducibility.
        """
        self.val_split = val_split
        self.download = download
        self.seed = seed

        self.transform = transforms.Compose(
            [
                transforms.ToImage(),
                transforms.ToDtype(torch.float32, scale=True),
            ]
        )

        self._load_datasets()

    def _load_datasets(self):
        # Load TinyImageNet splits and create a reproducible validation subset
        self.train_data = TinyImageNet(
            split="train", download=self.download, transform=self.transform
        )
        self.test_data = TinyImageNet(
            split="test", download=self.download, transform=self.transform
        )

        val_size = int(self.val_split * len(self.train_data))
        train_size = len(self.train_data) - val_size

        generator = torch.Generator().manual_seed(self.seed)
        self.train_subset, self.val_subset = torch.utils.data.random_split(
            self.train_data, [train_size, val_size], generator=generator
        )

    def get_dataloaders(self, batch_size=32, num_workers=3):
        """
        Build and return train, val, and test DataLoaders.
        """
        generator_train = torch.Generator().manual_seed(self.seed)

        train_loader = torch.utils.data.DataLoader(
            self.train_subset,
            batch_size=batch_size,
            shuffle=True,
            pin_memory=True,
            persistent_workers=False,
            num_workers=num_workers,
            generator=generator_train,
        )

        generator_val = torch.Generator().manual_seed(self.seed)
        val_loader = torch.utils.data.DataLoader(
            self.val_subset,
            batch_size=batch_size,
            shuffle=False,
            pin_memory=True,
            persistent_workers=False,
            num_workers=num_workers,
            generator=generator_val,
        )

        generator_test = torch.Generator().manual_seed(self.seed)
        test_loader = torch.utils.data.DataLoader(
            self.test_data,
            batch_size=batch_size,
            shuffle=False,
            pin_memory=True,
            persistent_workers=False,
            num_workers=num_workers,
            generator=generator_test,
        )

        return train_loader, val_loader, test_loader

    def __len__(self):
        return len(self.train_data)

    @property
    def num_classes(self):
        """
        Number of classes in the training set.
        Falls back to 200 if the dataset doesn't expose a classes attribute.
        """
        return (
            len(self.train_data.classes) if hasattr(self.train_data, "classes") else 200
        )

if __name__ == "__main__":
    # Quick sanity check: print dataset sizes
    dataset = TinyImageNetDataset(val_split=0.2, download=True, seed=42)
    train_loader, val_loader, test_loader = dataset.get_dataloaders(batch_size=64)

    print(f"Number of training samples: {len(train_loader.dataset)}")
    print(f"Number of validation samples: {len(val_loader.dataset)}")
    print(f"Number of test samples: {len(test_loader.dataset)}")