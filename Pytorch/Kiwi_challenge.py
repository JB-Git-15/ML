
import os
import sys
import requests
import zipfile
from pathlib import Path

import torchvision
from torchvision import transforms

import torch
import torch.nn as nn
from ml_functions.data_loaders.create_data_loaders import create_data_loaders
from ml_functions.train_test.Train_Test import train_test_model
from ml_functions.model_builders.Model_builder import TinyVGG


# load the data
train_dir = './data/pizza_steak_sushi/train/'
test_dir = './data/pizza_steak_sushi/test/'


transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.Grayscale(),
    transforms.ToTensor()
])

batch_size = 32
train_dataloader, test_dataloader, class_names = create_data_loaders(
    train_dir, test_dir, transform, batch_size)

# set device
device = "cuda" if torch.cuda.is_available() else "cpu"

# load the model
tiny_vgg = TinyVGG(input_shape=1, hidden_units=10, output_shape=3)

n_epochs = 25
n_samples = 5
lr = 0.1

loss_fn = nn.modules.loss.BCELoss()
optimizer = torch.optim.SGD(params=tiny_vgg.parameters(), lr=lr)
n_classes = len(class_names)
# train_test_model

loss_train_epochs, acc_train_epochs, loss_test_epochs, acc_test_epochs, num_epoch = train_test_model(
    n_epochs, n_samples, len(class_names), tiny_vgg, loss_fn, optimizer, train_dataloader, test_dataloader, device)


"""
# Setup path to data folder
data_path = Path("../data/")
image_path = data_path / "pizza_steak_sushi"

# If the image folder doesn't exist, download it and prepare it... 
if image_path.is_dir():
    print(f"{image_path} directory exists.")
else:
    print(f"Did not find {image_path} directory, creating one...")
    image_path.mkdir(parents=True, exist_ok=True)

# Download pizza, steak, sushi data
with open(data_path / "pizza_steak_sushi.zip", "wb") as f:
    request = requests.get("https://github.com/mrdbourke/pytorch-deep-learning/raw/main/data/pizza_steak_sushi.zip")
    print("Downloading pizza, steak, sushi data...")
    f.write(request.content)

# Unzip pizza, steak, sushi data
with zipfile.ZipFile(data_path / "pizza_steak_sushi.zip", "r") as zip_ref:
    print("Unzipping pizza, steak, sushi data...") 
    zip_ref.extractall(image_path)

# Remove zip file
os.remove(data_path / "pizza_steak_sushi.zip")
"""
