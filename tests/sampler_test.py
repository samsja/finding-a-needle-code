from src.few_shot_learning import RelationNet,sampler
from src.few_shot_learning.datasets import FewShotDataSets 

import torch

import unittest


class TestRelationNet(unittest.TestCase):
    def setUp(self):

        self.ep = 10
        self.n = 2
        self.k = 5
        self.q = 15
        self.f_dim = 64

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.FewShotSampler(
            bg_dataset,
            number_of_batch=2,
            episodes=self.ep,
            sample_per_class=self.n,
            classes_per_ep=self.k,
            queries=self.q,
        )

    def test_output_shape(self):

        assert self.model(self.inputs, self.ep, self.n, self.k, self.q).shape == (
            self.ep,
            self.k,
            self.k,
            self.q,
        )
