import torchvision
import torch
import os

from .datasets import FewShotDataSet
from torchvision import transforms
from PIL import Image
from tqdm import tqdm



class TrafficSignDataset(FewShotDataSet):
    def __init__(self, root_dir, exclude_class):
        super(TrafficSignDataset, self).__init__()
        """
        Args:
            root_dir : string. Directory with all the folders.
            exclude_class : list. List containing indices of which classes 
                                  to exclude from the training set.

        """

        self.data = []
        self.labels = []
        self.class_dir = {}

        c_idx = 0
        for i, c in tqdm(enumerate(os.listdir(root_dir))):
            if i not in exclude_class:
                list_of_datapoints = os.listdir(root_dir+"/"+c)

                self.data += list_of_datapoints
                self.labels += [c_idx]*len(list_of_datapoints)
                self.class_dir[c_idx] = root_dir + "/" + c
                
                c_idx += 1

        self._classes = torch.tensor(self.labels).unique()

        self.transform = transforms.Compose([
                                            transforms.Resize(256),
                                            transforms.RandomCrop(224),
                                            transforms.ToTensor(),
                                            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                 std=[0.229, 0.224, 0.225])
                                           ])

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
        url = self.class_dir[y] + "/" + self.data[idx]
        
        x = Image.open(url)
        x = self.transform(x)

        data = {"img" : x,
                "label" : torch.tensor(y),
                "file_name" : self.data[idx]}
        
        return data


    def add_datapoint(self, file_name, c_idx):
        """
        Method for adding a single data point to the dataset.

        # Args:
            file_name : String. Name of file that should be added to dataset.
            c_idx : int. Class index of which the file belongs to.

        """

        try:
            i = self.labels.index(c_idx)
        except:
            i = len(self.labels)
            
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

        del self.data[i]
        del self.labels[i]

        self._classes = torch.tensor(self.labels).unique()