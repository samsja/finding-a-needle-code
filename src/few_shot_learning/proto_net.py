import torch.nn as nn
import torch.nn.functional as F
import torch

from torchvision.models import resnet18


class ProtoNet(torch.nn.Module):
    """
    Implementation of the RelationNet model in pytorch

    # Arguments
        in_channels: int. number of input channels
        debug: bool. if true debug mode activate defaulf  False
    """

    def __init__(self, in_channels: int, debug: bool = False):
        super().__init__()

        self.debug = debug

        resnet = list(resnet18().children())[:-1]
        self.feature_extractor = nn.Sequential(*resnet)


    def forward(self, x):
        return self.feature_extractor(x).view(-1, 512)

    def _get_features_support(self, support: torch.Tensor) -> torch.Tensor:
        """
        project support image to embedded space then sum the features with in each class

        #Arguments:
            support: torch.tensor. data for the support
        """

        c = self.feature_extractor(support).view(-1, 512)
        return c.mean(dim=0)