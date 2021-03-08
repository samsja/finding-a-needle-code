import torch
import torchvision
from torchvision.datasets.utils import list_files

import os
from os.path import join
from typing import Any, Callable, List, Optional, Tuple
from .datasets import FewShotDataSet
from PIL import Image


def list_dir(root: str, prefix: bool = False) -> List[str]:
    """List all directories at a given root
    Args:
        root (str): Path to directory whose folders need to be listed
        prefix (bool, optional): If true, prepends the path to each result, otherwise
            only returns the name of the directories found
    """
    root = os.path.expanduser(root)
    directories = [p for p in os.scandir(root) if os.path.isdir(os.path.join(root, p))]
    if prefix is True:
        directories = [os.path.join(root, d) for d in directories]
    return directories


class MiniImageNet(FewShotDataSet):
    def __init__(
        self, path: str, validation: bool = False, transform: Optional[Callable] = None
    ):
        """
        DataSet for the mini image net

        #Args:

            path: str. path to the data
            validation: bool. If False will use the train folder
            max_len: int. the length of one class (i.e folder)
            transform (callable, optional): A function/transform that  takes in an PIL image
                and returns a transformed version. E.g, ``transforms.RandomCrop``
        """

        super(MiniImageNet, self).__init__()

        self.path = path + "/val" if validation else path + "/train"
        self.transform = transform

        self._classes_path = list_dir(self.path)

        self._classes = torch.arange(len(self._classes_path)).type(torch.long)

        self._classe_images = [
            [(image, idx) for image in list_files(join(self.path, classe), ".JPEG")]
            for idx, classe in enumerate(self._classes_path)
        ]

        self._flat_classe_images: List[Tuple[str, int]] = sum(self._classe_images, [])

        self._length_of_class = len(self._classe_images[0])

    def __len__(self):
        return len(self._flat_classe_images)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target classe.
        """

        image_name, classe = self._flat_classe_images[index]
        image_path = join(self.path, self._classes_path[classe], image_name)
        image = Image.open(image_path, mode="r").convert("L")

        if self.transform:
            image = self.transform(image)

        return image, classe

    def get_index_in_class(self, class_idx: int):
        """
        Method to get the indexes of the elements in the same class as class_idx

        # Args:
            class_idx : int. The index of the desired class

        """
        return torch.arange(
            class_idx * self._length_of_class,
            (class_idx + 1) * self._length_of_class,
            dtype=torch.long,
        )
