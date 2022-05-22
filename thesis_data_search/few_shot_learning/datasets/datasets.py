import torch
from typing import List


class FewShotDataSet(torch.utils.data.Dataset):
    """
    Pytorch DataSet that work for few shot dataset sampler

    #Attribute:
        classes: List. list of all the possible classes
    """

    def __init__(self, *argv):
        super(torch.utils.data.Dataset, self).__init__(*argv)
        self._classes: torch.Tensor[torch.int] = torch.Tensor([])

    @property
    def classes(self):
        return self._classes

    def get_index_in_class(self, class_idx: int):
        """
        Method to get the indexes of the elements in the same class as class_idx

        # Args:
            class_idx : int. The index of the desider class

        """
        raise NotImplementedError

    def get_index_in_class_vect(self, class_idx: List[int]):
        """
        Method to get the indexes of the elements in the same class as class_idx

        # Args:
            class_idx : list of index The index of the desider class

        """
        raise NotImplementedError
