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
from torchvision.datasets import CIFAR100
from torchvision.utils import save_image

os.chdir("..")

from thesis_data_search.few_shot_learning.datasets.traffic_signs import (
    TrafficSignDataset,
)

transform = torchvision.transforms.Compose(
    [
        torchvision.transforms.ToTensor(),
    ]
)

dataset_train_eval = CIFAR100(
    root="data", train=True, transform=transform, download=True
)

dataset_test = CIFAR100(root="data", train=False, transform=transform, download=True)

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


list_train = dataset_to_files(dataset_train, "data/cifar_100", "train")

with open("thesis_data_search/pickles/cifar_100/train.pkl", "wb") as f:
    pickle.dump(list_train, f)

list_eval = dataset_to_files(dataset_eval, "data/cifar_100", "eval")
list_test = dataset_to_files(dataset_test, "data/cifar_100", "test")

with open("thesis_data_search/pickles/cifar_100/test.pkl", "wb") as f:
    pickle.dump(list_test, f)
with open("thesis_data_search/pickles/cifar_100/eval.pkl", "wb") as f:
    pickle.dump(list_eval, f)

with open("thesis_data_search/pickles/cifar_100/label_list.pkl", "wb") as f:
    pickle.dump(dataset_train_eval.classes, f)
