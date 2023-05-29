from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os


def create_data_loaders(train_dir: str, test_dir: str, transform: transforms.Compose, batch_size: int):

    train_set = datasets.ImageFolder(train_dir, transform=transform)
    test_set = datasets.ImageFolder(test_dir, transform=transform)

    train_dataloader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(
        test_set, batch_size=batch_size, shuffle=False)
    class_names = train_set.classes

    return train_dataloader, test_dataloader, class_names
