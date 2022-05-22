from thesis_data_search.few_shot_learning import RelationNet
import torch

import unittest
import pytest


class TestRelationNet(unittest.TestCase):
    def setUp(self):

        self.ep = 10
        self.n = 2
        self.k = 5
        self.q = 15
        self.f_dim = 64

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.inputs = torch.rand(
            self.ep * self.k * (self.n + self.q), 1, 28, 28, device=self.device
        )

        self.model = RelationNet(
            in_channels=1, out_channels=self.f_dim, device=self.device, debug=True
        )

    def test_output_shape(self):

        assert self.model(self.inputs, self.ep, self.n, self.k, self.q).shape == (
            self.ep,
            self.k,
            self.k,
            self.q,
        )

    def test_concat(self):

        features_queries = torch.rand(self.ep * self.k, self.q, self.f_dim, 5, 5)
        features_supports = torch.rand(self.ep * self.k, self.f_dim, 5, 5)

        features_cat, _ = self.model._concat_features(
            features_supports, features_queries, self.ep, self.n, self.k, self.q
        )

        assert features_cat.shape == (
            self.ep,
            self.k,
            self.k,
            self.q,
            2 * self.f_dim,
            5,
            5,
        )
