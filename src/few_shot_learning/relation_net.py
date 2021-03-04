import torch
import torch.nn as nn
from typing import List, Tuple, Any
import torch.nn.functional as F


def get_conv_block_mp(
    in_channels: int, out_channels: int, padding: int = 0
) -> nn.Module:
    """
    Returns a Module that performs 3x3 convolution, ReLu activation, 2x2 max pooling.

    # Arguments
        in_channels:
        out_channels:
        padding:
    """

    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=padding),
        nn.BatchNorm2d(out_channels, affine=True, momentum=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2),
    )


def get_conv_block(in_channels: int, out_channels: int, padding: int = 1) -> nn.Module:
    """
    Returns a Module that performs 3x3 convolution, ReLu activatio.

    # Arguments
        in_channels:
        out_channels:
    """

    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=padding),
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
    def __init__(self, input_size: int, hidden_size: int = 8, adaptative_size: int = 2):
        """
        Basic BasicRelationModule for the RelationNet
        # Arguments
            input_size: int. feature space dimension
            hidden_size: int. size of the hidden layer
            adaptative_size : int. size of the
        """

        super(BasicRelationModule, self).__init__()

        self.conv1 = get_conv_block_mp(input_size * 2, input_size, padding=1)

        #  self.conv2 = get_conv_block(input_size, input_size, padding=1)
        #
        #  self.adapt = nn.AdaptiveMaxPool2d(adaptative_size)
        #
        self.conv2 = get_conv_block_mp(input_size, input_size, padding=1)
        self.linear1 = nn.Linear(input_size * adaptative_size ** 2, hidden_size)
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        forward method
        """
        x = self.conv1(x)

        x = self.conv2(x)
        # x = self.adapt(x)
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
        device: (torch.device)
        relation_module: (torch.nn.Module)
        embedding_module: (torch.nn.Module)
    """

    def __init__(
        self,
        device: torch.device,
        in_channels: int = None,
        out_channels: int = None,
        debug: bool = False,
        embedding_module: torch.nn.Module = None,
        relation_module: torch.nn.Module = None,
    ):
        super().__init__()

        if embedding_module is None:
            if in_channels is None or out_channels is None:
                raise ValueError(
                    "In channel and out channel are needed for embedding_module"
                )
            self.embedding = BasicEmbeddingModule(in_channels, out_channels)
        else:
            self.embedding = embedding_module

        if relation_module is None:
            if out_channels is None:
                raise ValueError(" out channel is needed for relation_module")

            self.relation = BasicRelationModule(out_channels)
        else:
            self.relation = relation_module

        self.debug = debug
        self.device = device

    def _concat_features(
        self,
        features_supports: torch.Tensor,
        features_queries: torch.Tensor,
        episodes: int,
        sample_per_class: int,
        classes_per_ep: int,
        queries: int,
    ) -> torch.Tensor:
        """
        concat feature
        #Arguments
            feature_support: torch.Tensor. feature for the support
            feature_queries: torch.Tensor. featur for the querie
        """

        # expand without copy memory for the concatenation and cat -> [ep,k,k,q,2*x,y,y]

        features_supports = features_supports.view(
            episodes, classes_per_ep, 1, 1, *features_supports.shape[-3:]
        )

        features_queries = features_queries.view(
            episodes, 1, classes_per_ep, *features_queries.shape[-4:]
        )

        features_supports = features_supports.expand(
            *features_supports.shape[:2],
            classes_per_ep,
            queries,
            *features_supports.shape[-3:]
        )

        features_queries = features_queries.expand(
            features_queries.shape[0], classes_per_ep, *features_queries.shape[-5:]
        )

        features_cat = torch.cat((features_supports, features_queries), dim=4)

        if self.debug:
            pass

        return features_cat

    def forward(
        self,
        inputs: torch.Tensor,
        episodes: int,
        sample_per_class: int,
        classes_per_ep: int,
        queries: int,
    ) -> torch.Tensor:
        """
        forward pass for the relation net

        #Arguments
            support: torch.tensor. data for the support
            queries: torch.Tensor. data for the queries

        # Return:
            relation_table: torch.tensor. R_i_j is the relation score between queries i and class j
        """

        # appply embedding
        features = self.embedding(inputs)

        features = features.view(
            episodes * classes_per_ep, sample_per_class + queries, *features.shape[-3:]
        )
        features_supports, features_queries = (
            features[:, :sample_per_class],
            features[:, -queries:],
        )

        # sum features over each sample per class
        features_supports = features_supports.sum(dim=1)

        features_cat = self._concat_features(
            features_supports,
            features_queries,
            episodes,
            sample_per_class,
            classes_per_ep,
            queries,
        )

        features_cat_shape = features_cat.shape

        features_cat = features_cat.view(-1, *features_cat.shape[-3:])

        relation = self.relation(features_cat)

        return relation.view(*features_cat_shape[:4])
