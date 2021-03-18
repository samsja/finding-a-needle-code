import torch
from typing import Iterator, List, Tuple, Any
from .datasets import FewShotDataSet


class FewShotSampler(torch.utils.data.Sampler):
    def __init__(
        self,
        dataset: FewShotDataSet,
        number_of_batch: int = 1,
        episodes: int = 1,
        sample_per_class: int = 1,
        classes_per_ep: int = 1,
        queries: int = 1,
    ):

        """Pytorch sampler to generates batches

        # Argumemts:
            dataset: torch.utils.data.Dataset the dataset on which the sampler will work
            number_of_batch: int , size of the batch
            episodes: int. number of episodes of n-shot,k-way,q-queries for the batch
            sample_per_class: int. Number of sample of each class
            class_it: int. Number of classes in the episode
            queries: number of queries for each selected class
        """

        super(FewShotSampler, self).__init__(dataset)

        self.dataset = dataset

        self.sample_per_class = sample_per_class
        self.classes_per_ep = classes_per_ep
        self.queries = queries
        self.episodes = episodes
        self.number_of_batch = number_of_batch

    def __len__(self) -> int:
        """
        len method needed to define a sampler

        # Return
            the lenght of the batch
        """

        return self.number_of_batch

    def __iter__(self) -> Iterator:

        """
        iterator method needed for the sampler

        # yield:
            return the index of the episodes batches compose of episodes with each time a support and queries, the indexes are reshape in a (X,1) tensor
        """

        index_to_yield: torch.Tensor = torch.zeros(
            (self.episodes, self.classes_per_ep, self.sample_per_class + self.queries),
            dtype=torch.int,
        )

        for batch in range(self.number_of_batch):
            for episode in range(self.episodes):

                classes_for_ep: torch.Tensor[torch.int] = self.dataset.classes[
                    torch.multinomial(
                        torch.ones(len(self.dataset.classes)),
                        num_samples=self.classes_per_ep,
                        replacement=False,
                    )
                ]

                for i, c in enumerate(classes_for_ep):
                    samples_same_class = self.dataset.get_index_in_class(c)
                    index_to_yield[episode, i] = samples_same_class[
                        torch.randint(
                            0,
                            len(samples_same_class),
                            (self.sample_per_class + self.queries,),
                            dtype=torch.long,
                        )
                    ]

            yield index_to_yield.view(-1, 1)
