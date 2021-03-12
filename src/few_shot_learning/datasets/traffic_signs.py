import torchvision
import random
import torch
import os

from datasets import FewShotDataSet
from torchvision import transforms
from PIL import Image
from tqdm import tqdm



class TrafficSignDataset(FewShotDataSet):
    def __init__(self, root_dir, exclude_class, transform):
        super(TrafficSignDataset, self).__init__()
        """
        Args:
            root_dir : string. Directory with all the folders.
            exclude_class : list. List containing indices of which classes 
                                  to exclude from the training set.

        """

        self.data = []
        self.labels = []
        self.labels_str = []
        
        c_idx = 0

        for i, c in tqdm(enumerate(os.listdir(root_dir))):
            if i not in exclude_class:
                list_of_datapoints = os.listdir(root_dir + "/" + c)
                list_of_datapoints = [root_dir + "/" + c + "/" + s for s in list_of_datapoints]
                
                self.data += list_of_datapoints
                self.labels += [c_idx]*len(list_of_datapoints)
                self.labels_str.append(c)
                
                c_idx += 1

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

        if label not in self.labels:
            del self.labels_str[label] 

        self._classes = torch.tensor(self.labels).unique()