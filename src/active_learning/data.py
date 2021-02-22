from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from random import sample, shuffle
from skimage import io

import os
import torch


class LabeledDataset(Dataset):
    """Labeled dataset."""

    def __init__(self, root_dir: str, k: int):
        """
        Args:
            root_dir (string): Directory with all the folders.
            k (int): Number of samples to be used for each class
        """
        self.data = {}
        self.unlabeled = {}
        self.root = root_dir

        for c in os.listdir(root_dir):
            self.data[c] = []
            self.unlabeled[c] = []

            list_of_datapoints = os.listdir(root_dir + "/" + c)
            shuffle(list_of_datapoints)

            self.data[c] = list_of_datapoints[:k]
            self.unlabeled[c] = list_of_datapoints[k:]

    def __len__(self):
        return len(self.data.keys())

    def __getitem__(self, idx):
        c = self.data.keys()[idx]
        tmp = sample(self.data[c], 6)

        q = torch.tensor(io.imread(self.root + "/" + c + "/" + tmp[0]))

        s = []
        for fn in tmp[1:]:
            s.append(io.imread(self.root + "/" + c + "/" + fn))

        s = torch.tensor(s)

        return q, s

    def get_support(self, c: str):
        tmp = sample(self.data[c], 5)

        s = []
        for fn in tmp[1:]:
            s.append(io.imread(fn))

        s = torch.tensor(s)

        return s

    def add_datapoints(self, datapoints: list):
        for datapoint in datapoints:
            fn, c = datapoint[0], datapoint[1]
            self.data[c].append(fn)


class UnlabeledDataset(Dataset):
    """Unlabeled dataset."""

    def __init__(self, data_pool: dict):
        """
        Args:
            data_pool (string): Dictionary mapping  filenames to classes
        """
        self.data = {}

        for c in data_pool:
            for fn in c.values():
                self.data[fn] = c

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        fn = self.data.values()[idx]
        c = self.data[fn]

        q = torch.tensor(io.imread(self.root + "/" + c + "/" + fn))

        x = {"image": q, "fn": fn, "label": c}

        return x

    def remove_dapoints(self, datapoints: list):
        for datapoint in datapoints:
            del self.data[datapoint[0]]
