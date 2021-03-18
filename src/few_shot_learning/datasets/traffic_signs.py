import torchvision
import random
import torch
import os

from .datasets import FewShotDataSet
from torchvision import transforms
from PIL import Image
from tqdm import tqdm



class TrafficSignDataset(FewShotDataSet):
    def __init__(self, file_names, transform):
        super(TrafficSignDataset, self).__init__()
        """
        Args:
            file_names : list. List containg file names of all images that should be added.

        """

        self.data = []
        self.labels = []
        self.labels_str = []

        for fn in tqdm(file_names):
            self.data.append(fn)

            label = fn.split("/")[3]

            if label not in self.labels_str:
                self.labels_str.append(label)

            self.labels.append(self.labels_str.index(label))
                

        self._classes = torch.tensor(self.labels).unique()

        self.transform = transform


    def __len__(self):
        return len(self.labels)


    def get_index_in_class(self, class_idx):
        """
        Method to get the indexes of the elements in the same class as class_idx

        # Args:
            class_idx : int. The index of the desider class

        """

        start = self.labels.index(class_idx)

        try:
            end = self.labels.index(class_idx+1)
        except:
            end = len(self.labels)

        return torch.arange(start, end)


    def __getitem__(self, idx):
        y = self.labels[idx]
        
        url = self.data[idx]
        
        x = Image.open(url)
        x = self.transform(x)

        data = {"img" : x,
                "label" : torch.tensor(y),
                "file_name" : self.data[idx]}
        
        return data

    def get_support(self, n, k):
        indices = self.get_index_in_class(k)
        
        samples = random.sample(n, indices)

        batch = []
        
        for i in samples:
            x = self.__get_item(i)["img"]
            batch.append(x)

        return torch.tensor(batch)


    def get_all_support_set(self, k):
        batch = []
        
        for c in range(self._classes):
            indices = self.get_index_in_class(c)
            samples = random.sample(k, indices)
            
            for i in samples:
                x = self.__get_item(i)["img"]
                batch.append(x)

        return torch.tensor(batch)

    
    def add_datapoint(self, file_name, class_):

        """
        Method for adding a single data point to the dataset.

        # Args:
            file_name : String. Name of file that should be added to dataset.
            c_idx : str. Name of class data point belongs to

        """

        if class_ in self.labels_str:
            c_idx = self.labels_str.index(class_) # Get class index
            i = self.labels.index(c_idx) # Find first position of the class

        else:
            i = len(self.labels)
            c_idx = len(self.labels_str)
            
            self.labels_str.append(class_)
            
        self.data.insert(i, file_name)
        self.labels.insert(i, c_idx)

        self._classes = torch.tensor(self.labels).unique()


    def remove_datapoint(self, file_name):
        """
        Method for removing single data point from dataset

        # Args:
            file_name : String. Name of the file that should be removed.

        """

        i = self.data.index(file_name)
        label = self.labels[i]

        del self.data[i]
        del self.labels[i]


        self._classes = torch.tensor(self.labels).unique()
