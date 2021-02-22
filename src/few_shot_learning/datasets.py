import torch


class FewShotDataSet(torch.utils.data.Dataset):
    """
    Pytorch DataSet that work for few shot dataset sampler

    #Attribute:
        classes: List. list of all the possible classes
    """

    def __init__(self, *argv):
        super(torch.utils.data.Dataset, self).__init__(*argv)
        self.classes: torch.Tensor[torch.int] = torch.Tensor([])

    def get_index_in_class(self, class_idx: int):
        """
        Method to get the indexes of the elements in the same class as class_idx

        # Args:
            class_idx : int. The index of the desider class

        """
        raise NotImplementedError
