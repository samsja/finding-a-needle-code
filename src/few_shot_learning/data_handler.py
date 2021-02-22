import torch
from typing import Iterator, List, Tuple, Any
from PIL import Image


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


class FewShotSampler(torch.utils.data.Sampler):
    def __init__(
        self,
        dataset: FewShotDataSet,
        sample_per_class: int = 1,
        classes_per_ep: int = 1,
        queries: int = 1,
        episodes: int = 1,
        batch_size: int = 1,
    ):

        """Pytorch sampler to generates batches

        # Argumemts:
            dataset:
            sample_per_class: int. Number of sample of each class
            class_it: int. Number of classes in the episode
            queries: number of queries for each selected class
            episodes: int. number of episodes of n-shot,k-way,q-queries for the batch
            batch_size: int, the size of the batch with episodes
        """

        super(FewShotSampler, self).__init__(dataset)

        self.dataset = dataset

        self.sample_per_class = sample_per_class
        self.classes_per_ep = classes_per_ep
        self.queries = queries
        self.episodes = episodes
        self.batch_size = batch_size

    def __len__(self) -> int:
        """
        len method needed to define a sampler

        # Return
            the lenght of the batch
        """

        return self.episode

    def __iter__(self) -> Iterator:
        
        """
        iterator method needed for the sampler
        
        # yield:
            return the index of the batch_size batches compose of episodes with each time a support and queries, the indexes are reshape in a (X,1) tensor
        """

        index_to_yield: torch.Tensor[[torch.int, torch.int, torch.int]] = torch.zeros(
            (self.batch_size, self.classes_per_ep, self.sample_per_class),
            dtype=torch.int,
        )

        for batch in range(self.batch_size):

            for _ in range(self.episodes):

                classes_for_ep: torch.Tensor[torch.int] = self.dataset.classes[
                    torch.randint(0, len(self.dataset.classes), (self.classes_per_ep,))
                ]

                for i, c in enumerate(classes_for_ep):
                    samples_same_class = self.dataset.get_index_in_class(c)
                    index_to_yield[batch, i] = samples_same_class[
                        torch.randint(
                            0,
                            len(samples_same_class),
                            (self.sample_per_class,),
                            dtype=torch.long,
                        )
                    ]

        yield index_to_yield.view(-1, 1)
