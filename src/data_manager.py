import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

from src.paths import Paths


class DataManager:
    def __init__(self, root=Paths.PROJECT_ROOT):
        self.root = root

        self.class_names = {
            "mnist": {i: str(i) for i in range(10)},
            "cifar10": {
                0: "airplane",
                1: "automobile",
                2: "bird",
                3: "cat",
                4: "deer",
                5: "dog",
                6: "frog",
                7: "horse",
                8: "ship",
                9: "truck",
            },
        }

    def get_class_names(self, dataset_name):
        dataset_name = dataset_name.lower()

        if dataset_name not in self.class_names:
            raise ValueError(f"Unknown dataset: {dataset_name}")

        return self.class_names[dataset_name]

    def get_transform(self, dataset_name):
        dataset_name = dataset_name.lower()

        if dataset_name == "mnist":
            return transforms.Compose(
                [
                    transforms.Resize((224, 224)),
                    transforms.Grayscale(num_output_channels=3),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5,), (0.5,)),
                ]
            )

        if dataset_name == "cifar10":
            return transforms.Compose(
                [
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        (0.5, 0.5, 0.5),
                        (0.5, 0.5, 0.5),
                    ),
                ]
            )

        raise ValueError(f"Unknown dataset: {dataset_name}")

    def get_dataset(self, dataset_name, train=True, download=True):
        dataset_name = dataset_name.lower()
        transform = self.get_transform(dataset_name)

        if dataset_name == "mnist":
            return datasets.MNIST(
                root=self.root,
                train=train,
                download=download,
                transform=transform,
            )

        if dataset_name == "cifar10":
            return datasets.CIFAR10(
                root=self.root,
                train=train,
                download=download,
                transform=transform,
            )

        raise ValueError(f"Unknown dataset: {dataset_name}")

    def get_train_val_datasets(
        self,
        dataset_name,
        val_ratio=0.1,
        download=True,
        seed=42,
    ):
        full_train_dataset = self.get_dataset(
            dataset_name=dataset_name,
            train=True,
            download=download,
        )

        val_size = int(len(full_train_dataset) * val_ratio)
        train_size = len(full_train_dataset) - val_size

        generator = torch.Generator().manual_seed(seed)

        train_dataset, val_dataset = random_split(
            full_train_dataset,
            [train_size, val_size],
            generator=generator,
        )

        return train_dataset, val_dataset

    def get_test_dataset(self, dataset_name, download=True):
        return self.get_dataset(
            dataset_name=dataset_name,
            train=False,
            download=download,
        )

    def get_data_loaders(
        self,
        dataset_name,
        batch_size=64,
        val_ratio=0.1,
        num_workers=4,
        pin_memory=True,
        download=True,
        seed=42,
    ):
        train_dataset, val_dataset = self.get_train_val_datasets(
            dataset_name=dataset_name,
            val_ratio=val_ratio,
            download=download,
            seed=seed,
        )

        test_dataset = self.get_test_dataset(
            dataset_name=dataset_name,
            train=False,
            download=download,
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

        return train_loader, val_loader, test_loader
