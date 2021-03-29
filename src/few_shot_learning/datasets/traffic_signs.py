import torchvision
import random
import torch
import glob
import os

from .datasets import FewShotDataSet
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
from typing import List

def get_file_name_from_folder(root_dir, exclude_class):

    list_of_datapoints = []
    label_list = []

    for i, c in tqdm(enumerate(os.listdir(root_dir))):
        if i not in exclude_class:
            list_of_datapoints_in_folder = os.listdir(root_dir + "/" + c)
            list_of_datapoints += [
                root_dir + "/" + c + "/" + s for s in list_of_datapoints_in_folder
            ]

            label_list.append(c)

    return list_of_datapoints,label_list


class TrafficSignDataset(FewShotDataSet):
    def __init__(self, file_names : List[str], label_list : List[str], transform, root_dir : str,exclude_class: List[int] = []):
        super().__init__()
        """
        Args:
            file_names : list. List containg file names of all images that should be added.
            label_list : list. List containg names of all classes.

        """

        self.data = []
        self.labels = []
        self.labels_str = label_list
        self.root_dir = root_dir

        self.classes_indexes = [[] for _ in label_list]

        for fn in tqdm(file_names):
            label = fn.split("/")[-2]
            
            label_idx = self.labels_str.index(label)
 
            if label_idx not in exclude_class:          
                self.labels.append(label_idx)
                self.data.append(f"{root_dir}/{fn}")

        
        self._unique_classes = torch.tensor(self.labels).unique()
        self._classes = torch.arange(self._unique_classes.size(0))


        self.update_classes_indexes()

        self.transform = transform


    def __len__(self):
        return len(self.labels)

    def update_classes_indexes(self):


        self.classes_indexes = [[] for _ in range(len(self.classes_indexes) )]
        for index, label_idx in enumerate(self.labels):
            self.classes_indexes[label_idx].append(index)

        for i in range(len(self.classes_indexes)):
            self.classes_indexes[i] = torch.tensor(self.classes_indexes[i])

    def get_index_in_class(self, class_idx):
        """
        DEPRECATED
        Method to get the indexes of the elements in the same class as class_idx

        # Args:
            class_idx : int. The index of the desider class

        """

        start = self.labels.index(class_idx)

        try:
            end = self.labels.index(class_idx + 1)
        except:
            end = len(self.labels)

        return torch.arange(start, end)

    def get_index_in_class_opt(self, class_idx):
        """
        Method to get the indexes of the elements in the same class as class_idx

        # Args:
            class_idx : int. The index of the desider class

        """

        return self.classes_indexes[self._unique_classes[class_idx]]

    def get_index_in_class_vect(self, class_idx_vect: torch.Tensor):
        """
        vectorized Method to get the indexes of the elements in the same class as class_idx

        # Args:
            class_idx : torch.Tensor[int]. The indexes of the desired classes

        """

        return _script_get_index_in_class_vect(class_idx_vect, self.classes_indexes,self._unique_classes)

    def __getitem__(self, idx):
        y = self.labels[idx]

        url = self.data[idx]

        x = Image.open(url)
        x = self.transform(x)

        if not(type(idx)==torch.Tensor):
            idx = torch.Tensor([idx]).clone().detach()

        data = {"img": x, 
                "label": torch.tensor(y), 
                "id": idx}

        return data


    def get_support(self, n, k):
        indices = self.get_index_in_class(k).tolist()

        samples = random.sample(indices, n)

        batch = []

        for i in samples:
            x = self.__getitem__(i)["img"]
            batch.append(x)

        return torch.stack(batch)

    def get_all_support_set(self, k):
        batch = []

        for c in range(self._classes):
            indices = self.get_index_in_class(c).tolist()

            samples = random.sample(indices, k)

            for i in samples:
                x = self.__getitem__(i)["img"]
                batch.append(x)

        return torch.stack(batch)

    def add_datapoint(self, file_name, class_):

        """
        Method for adding a single data point to the dataset.

        # Args:
            file_name : String. Name of file that should be added to dataset.
            class_ : String. Name of class data point belongs to

        """

        c_idx = self.labels_str.index(class_)  # Get class index

        try:
            i = self.labels.index(c_idx)  # Find first position of the class

        except:
            i = len(
                self.labels
            )  # If no data point of c_idx exists, add at end of list.

        self.data.insert(i, file_name)
        self.labels.insert(i, c_idx)
 
        self._unique_classes = torch.tensor(self.labels).unique()
        self._classes = torch.arange(self._unique_classes.size(0))

    def remove_datapoints(self, ids):
        """
        Method for removing single data point from dataset

        # Args:
            ids : list. Indices of the data point to remove.

        """

        self.data = [i for  j, i in enumerate(self.data) if j not in ids]

        self.labels = [i for  j, i in enumerate(self.labels) if j not in ids]
 
        self._unique_classes = torch.tensor(self.labels).unique()
        self._classes = torch.arange(self._unique_classes.size(0))

from typing import Dict, List


@torch.jit.script
def _script_get_index_in_class_vect(
    class_idx_vect: torch.Tensor, classes_indexes: List[torch.Tensor], unique_classes : torch.Tensor
):
    """
    vectorized Method to get the indexes of the elements in the same class as class_idx

    # Args:
        class_idx : torch.Tensor[int]. The indexes of the desired classes

    """

    return [classes_indexes[unique_classes[c].item()] for c in class_idx_vect]
