# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.11.3
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %load_ext autoreload
# %autoreload 2

import os
import pickle

import torch
import torchvision
from torch.utils.data import random_split
from torchvision.datasets import StanfordCars
from torchvision.utils import save_image

os.chdir("..")


transform = torchvision.transforms.Compose(
    [
        torchvision.transforms.ToTensor(),
    ]
)

dataset_train_eval = StanfordCars(
    root="data", split="train", transform=transform, download=True
)

dataset_test = StanfordCars(
    root="data", split="test", transform=transform, download=True
)

len(dataset_train_eval), len(dataset_test)

len_ = len(dataset_train_eval)
N = int(len_ * 0.9)
N, len_ - N

dataset_train, dataset_eval = random_split(
    dataset_train_eval, [N, len_ - N], generator=torch.Generator().manual_seed(42)
)

len(dataset_train), len(dataset_eval), len(dataset_test)


def init_path(path):
    os.makedirs(path, exist_ok=True)

    for class_ in dataset_train_eval.classes:
        os.makedirs(f"{path}/{class_}", exist_ok=True)


def dataset_to_files(dataset, path, name):

    init_path(path)
    list_files = []

    for i, (img, class_) in enumerate(dataset):
        file_name = f"{dataset_train_eval.classes[class_]}/{name}_{i}.png"
        full_path = f"{path}/{file_name}"

        save_image(img, full_path)
        list_files.append(file_name)

    return list_files


list_train = dataset_to_files(dataset_train, "data/cars", "train")

with open("thesis_data_search/pickles/cars/train.pkl", "wb") as f:
    pickle.dump(list_train, f)

list_eval = dataset_to_files(dataset_eval, "data/cars", "eval")
list_test = dataset_to_files(dataset_test, "data/cars", "test")

with open("thesis_data_search/pickles/cars/test.pkl", "wb") as f:
    pickle.dump(list_test, f)
with open("thesis_data_search/pickles/cars/eval.pkl", "wb") as f:
    pickle.dump(list_eval, f)

with open("thesis_data_search/pickles/cars/label_list.pkl", "wb") as f:
    pickle.dump(dataset_train_eval.classes, f)
