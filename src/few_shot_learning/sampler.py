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

        super().__init__(dataset)

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


# **********optim**************


class FewShotSampler2(torch.utils.data.Sampler):
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

        super().__init__(dataset)

        self.dataset = dataset

        self.sample_per_class = sample_per_class
        self.classes_per_ep = classes_per_ep
        self.queries = queries
        self.episodes = episodes
        self.number_of_batch = number_of_batch

        assert (self.dataset.classes == torch.arange(len(self.dataset.classes))).all()

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

        for batch in range(self.number_of_batch):

            index_to_yield = []
            classes_for_episodes = _class_for_episodes(
                self.episodes, len(self.dataset.classes), self.classes_per_ep
            )

            for episode in range(self.episodes):

                samples_same_class = self.dataset.get_index_in_class_vect(
                    classes_for_episodes[episode]
                )

                index_to_yield_ep = _prepare_one_ep(
                    self.classes_per_ep,
                    self.sample_per_class + self.queries,
                    samples_same_class,
                )

                index_to_yield.append(index_to_yield_ep)

            index_to_yield = torch.stack(index_to_yield, dim=0)

            yield index_to_yield.view(-1, 1)


@torch.jit.script
def _class_for_episodes(episodes: int, nb_classes: int, classes_per_ep: int):

    classes_for_ep = []

    for _ in range(episodes):
        classes_for_ep.append(
            torch.multinomial(
                torch.ones(nb_classes),
                num_samples=classes_per_ep,
                replacement=False,
            )
        )

    return torch.stack(classes_for_ep, dim=0)


@torch.jit.script
def _prepare_one_ep(
    classes_per_ep: int, sample_to_draw: int, samples_same_class: List[torch.Tensor]
):
    index_to_yield_ep = []
    for i in range(classes_per_ep):

        index_to_yield_ep.append(
            samples_same_class[i][
                torch.randint(
                    0,
                    len(samples_same_class[i]),
                    (sample_to_draw,),
                    dtype=torch.long,
                )
            ]
        )

    return torch.stack(index_to_yield_ep, dim=0)


#################


class ClassSampler(torch.utils.data.Sampler):
    """
    To sanpler data of a given classes of a FewShotDataSet
    """

    def __init__(self, dataset: FewShotDataSet, class_idx: int, batch_size: int):
        super().__init__(dataset)
        self.batch_size = batch_size
        self.dataset = dataset
        self.class_index = self.dataset.get_index_in_class(class_idx)

    def __len__(self) -> int:
        quotient = len(self.class_index) // self.batch_size
        rest = len(self.class_index) % self.batch_size

        return quotient + 1 if rest > 0 else quotient

    def __iter__(self) -> Iterator:
        for i in range(len(self)):
            yield self.class_index[
                i
                * self.batch_size : min(
                    (i + 1) * self.batch_size, len(self.class_index)
                )
            ]
