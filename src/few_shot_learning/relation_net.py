import torch
import torch.nn as nn
from typing import List, Tuple, Any
import torch.nn.functional as F


def get_conv_block_mp(in_channels: int, out_channels: int) -> nn.Module:
    """
    Returns a Module that performs 3x3 convolution, ReLu activation, 2x2 max pooling.

    # Arguments
        in_channels:
        out_channels:
    """

    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=0),
        nn.BatchNorm2d(out_channels, affine=True, momentum=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2),
    )


def get_conv_block(in_channels: int, out_channels: int) -> nn.Module:
    """
    Returns a Module that performs 3x3 convolution, ReLu activatio.

    # Arguments
        in_channels:
        out_channels:
    """

    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels, affine=True, momentum=1),
        nn.ReLU(),
    )


class BasicEmbeddingModule(nn.Module):
    """
    basic embedded Module used in the relation net models

    # Arguments
        in_channels: int
        out_channels: int
    """

    def __init__(self, in_channels: int, out_channels: int):
        super(BasicEmbeddingModule, self).__init__()

        list_conv_blocks = [
            get_conv_block_mp(in_channels, out_channels),
            get_conv_block_mp(out_channels, out_channels),
            get_conv_block(out_channels, out_channels),
            get_conv_block(out_channels, out_channels),
        ]

        self.conv_blocks = nn.Sequential(*list_conv_blocks)

    def forward(self, x):

        x = self.conv_blocks(x)

        return x


class BasicRelationModule(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 8):
        """
        Basic BasicRelationModule for the RelationNet
        # Arguments
            input_size: int. feature space dimension
            hidden_size: int. size of the hidden layer
        """

        super(BasicRelationModule, self).__init__()

        self.conv1 = get_conv_block_mp(input_size * 2, input_size)

        self.conv2 = get_conv_block_mp(input_size, input_size)

        self.linear1 = nn.Linear(input_size * 16, hidden_size)
        self.linear2 = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        forward method
        """
        x = self.conv1(x)
        x = self.conv2(x)

        x = x.view(x.size(0), -1)

        x = self.linear1(x)
        x = F.relu(x)
        x = self.linear2(x)
        x = torch.sigmoid(x)

        return x


class RelationNet(torch.nn.Module):
    """
    Implementation of the RelationNet model in pytorch

    # Arguments
        in_channels: int. number of input channels
        out_channels: int. number of input channels
        debug: bool. if true debug mode activate defaulf  False
    """

    def __init__(self, in_channels: int, out_channels: int, debug: bool = False):
        super().__init__()

        self.embedding = BasicEmbeddingModule(in_channels, out_channels)
        self.relation = BasicRelationModule(out_channels)
        self.debug = debug

    def _concat_features(
        self, feature_support: torch.Tensor, feature_queries: torch.Tensor
    ) -> torch.Tensor:
        """
        concat feature
        #Arguments
            feature_support: torch.Tensor. feature for the support
            feature_queries: torch.Tensor. featur for the querie
        """

        features_shape = list(feature_support.shape)
        features_shape[2] += feature_queries.size(1)
        features_shape[1] = feature_queries.size(0)

        features_cat = torch.zeros(features_shape)

        for i in range(features_cat.size(0)):
            for j in range(features_cat.size(1)):

                features_cat[i, j] = torch.cat(
                    (feature_support[i].squeeze(0), feature_queries[j]), 0
                )

        return features_cat

    def _get_features_support(self, support: torch.Tensor) -> torch.Tensor:
        """
        project support image to embedded space then sum the features with in each class

        #Arguments:
            support: torch.tensor. data for the support
        """

        feature_support = []
        for class_support in support:
            feature_support.append(self.embedding(class_support))

        feature_support = torch.stack(feature_support)
        feature_support = feature_support.sum(dim=1)
        feature_support = feature_support.unsqueeze(1)

        return feature_support

    def forward(self, support: torch.Tensor, queries: torch.Tensor) -> torch.Tensor:
        """
        forward pass for the relation net

        #Arguments
            support: torch.tensor. data for the support
            queries: torch.Tensor. data for the queries

        # Return:
            relation_table: torch.tensor. R_i_j is the relation score between queries i and class j
        """

        relation_table = torch.zeros(queries.size(1), queries.size(0), support.size(0))

        features_support = self._get_features_support(support)

        for i, queries_class in enumerate(queries):
            features_queries_class = self.embedding(queries_class)
            features_cat = self._concat_features(
                features_support, features_queries_class
            )

            for j in range(features_cat.size(1)):
                relation_table[j, i] = self.relation(features_cat[:, j]).flatten()
        return relation_table
