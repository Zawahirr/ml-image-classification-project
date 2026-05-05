from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


@dataclass(frozen=True)
class DatasetInfo:
    name: str
    num_classes: int
    input_shape: tuple[int, int, int]
    class_names: list[str]


def _fashion_mnist_transforms():
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    )


def _cifar10_transforms():
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        ]
    )


def load_image_dataset(
    dataset_name="fashion_mnist",
    batch_size=64,
    validation_split=0.1,
    data_dir="./data",
    num_workers=0,
    seed=42,
):
    """Load an image classification dataset with train/validation/test splits."""
    dataset_key = dataset_name.lower().replace("-", "_")

    if dataset_key == "fashion_mnist":
        transform = _fashion_mnist_transforms()
        train_dataset = datasets.FashionMNIST(
            root=data_dir,
            train=True,
            download=True,
            transform=transform,
        )
        test_dataset = datasets.FashionMNIST(
            root=data_dir,
            train=False,
            download=True,
            transform=transform,
        )
        info = DatasetInfo(
            name="Fashion-MNIST",
            num_classes=10,
            input_shape=(1, 28, 28),
            class_names=list(train_dataset.classes),
        )
    elif dataset_key == "cifar10":
        transform = _cifar10_transforms()
        train_dataset = datasets.CIFAR10(
            root=data_dir,
            train=True,
            download=True,
            transform=transform,
        )
        test_dataset = datasets.CIFAR10(
            root=data_dir,
            train=False,
            download=True,
            transform=transform,
        )
        info = DatasetInfo(
            name="CIFAR-10",
            num_classes=10,
            input_shape=(3, 32, 32),
            class_names=list(train_dataset.classes),
        )
    else:
        raise ValueError("dataset_name must be 'fashion_mnist' or 'cifar10'")

    val_size = int(len(train_dataset) * validation_split)
    train_size = len(train_dataset) - val_size
    generator = torch.Generator().manual_seed(seed)
    train_subset, val_subset = random_split(
        train_dataset,
        [train_size, val_size],
        generator=generator,
    )

    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    return train_loader, val_loader, test_loader, info


def load_fashion_mnist(batch_size=64):
    train_loader, _, test_loader, _ = load_image_dataset(
        dataset_name="fashion_mnist",
        batch_size=batch_size,
    )
    return train_loader, test_loader


if __name__ == "__main__":
    train_loader, val_loader, test_loader, info = load_image_dataset()

    print("Dataset:", info.name)
    print("Input shape:", info.input_shape)
    print("Train batches:", len(train_loader))
    print("Validation batches:", len(val_loader))
    print("Test batches:", len(test_loader))
