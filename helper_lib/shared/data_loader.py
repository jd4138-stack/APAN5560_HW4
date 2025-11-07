# helper_lib/data_loader.py
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


_CIFAR10_MEAN = [0.4914, 0.4822, 0.4465]
_CIFAR10_STD  = [0.2470, 0.2435, 0.2616]

def get_cifar10_loaders(
    data_root="datasets/cifar10",
    batch_size=128,
    image_size=64,          # upsample to match your CNN
    val_size=5000,          # split from the train set (50k -> 45k/5k)
    num_workers=2,
):
    train_tfms = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(_CIFAR10_MEAN, _CIFAR10_STD),
    ])
    test_tfms = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(_CIFAR10_MEAN, _CIFAR10_STD),
    ])

    full_train = datasets.CIFAR10(root=data_root, train=True, download=True, transform=train_tfms)
    test_set   = datasets.CIFAR10(root=data_root, train=False, download=True, transform=test_tfms)

    # train/val split
    train_size = len(full_train) - val_size
    train_set, val_set = random_split(full_train, [train_size, val_size],
                                      generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=torch.cuda.is_available())
    val_loader   = DataLoader(val_set, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=torch.cuda.is_available())
    test_loader  = DataLoader(test_set, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=torch.cuda.is_available())
    return train_loader, val_loader, test_loader

def get_mnist_loader(
    data_root="datasets/mnist",
    batch_size=128,
    train=True,
    num_workers=2,
):
    # Normalize to [-1, 1] because the Generator outputs with Tanh
    tfm = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])
    ds = datasets.MNIST(root=data_root, train=train, download=True, transform=tfm)
    return DataLoader(
        ds, batch_size=batch_size, shuffle=train, num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )

def get_data_loader(data_dir, batch_size=32, train=True):
    """
    Build a DataLoader for an ImageFolder directory.

    Args:
        data_dir (str): Path to a folder with class-per-subfolder structure.
        batch_size (int): Batch size for the loader.
        train (bool): If True, enables train-time shuffling and light augmentation.

    Returns:
        torch.utils.data.DataLoader: Data loader over the dataset at data_dir.
    """
    # Basic transforms for 64x64 RGB inputs
    tfms = [
        transforms.Resize((64, 64)),
    ]

    # Light augmentation for training
    if train:
        tfms.append(transforms.RandomHorizontalFlip())

    tfms.extend([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    transform = transforms.Compose(tfms)

    # Expecting class-per-subfolder structure
    dataset = datasets.ImageFolder(root=data_dir, transform=transform)

    # Shuffle only for training
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=True if (2 > 0) else False,
    )

    return loader
