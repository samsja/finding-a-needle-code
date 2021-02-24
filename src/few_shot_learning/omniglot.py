import torch
import torchvision

from .datasets import FewShotDataSet


class Omniglot(torchvision.datasets.Omniglot, FewShotDataSet):
    def __init__(self, *args, **kwargs):
        super(Omniglot, self).__init__(*args, **kwargs)

        self.classes: torch.Tensor[torch.int] = (
            torch.Tensor([label for (image_fn, label) in self._flat_character_images])
            .unique()
            .type(torch.long)
        )

        self.length_of_class = len(self._character_images[0])

    def get_index_in_class(self, class_idx: int):
        """
        Method to get the indexes of the elements in the same class as class_idx

        # Args:
            class_idx : int. The index of the desider class

        """
        return torch.arange(
            class_idx * self.length_of_class,
            (class_idx + 1) * self.length_of_class,
            dtype=torch.long,
        )
