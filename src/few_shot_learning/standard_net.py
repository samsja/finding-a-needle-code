import torch.nn as nn
import torch.nn.functional as F
import torch

from torchvision.models import resnet18
from .utils_train import ModuleAdaptater


class StandardNet(torch.nn.Module):
    """
    Implementation of the RelationNet model in pytorch

    # Arguments
        classes: int. Number of output nodes
    """

    def __init__(self, classes):
        super().__init__()

        resnet = list(resnet18().children())[:-1]
        
        self.feature_extractor = nn.Sequential(*resnet)
        self.fc = nn.Linear(512, classes)

    def forward(self, x):
        x = self.feature_extractor(x).view(-1, 512)

        return self.fc(x)



class StandardNetAdaptater(ModuleAdaptater):
    """
    Standard net module adaptater
    """

    def __init__(
        self,
        model: nn.Module,
        nb_ep: int,
        n: int,
        k: int,
        q: int,
        device: torch.device,
    ):
        super(StandardNetAdaptater, self).__init__(model)
        
        self.device = device
        self.model = model
        self.loss_fn = nn.CrossEntropyLoss()


    def get_loss_and_accuracy(self, inputs, labels, accuracy=False):
        x, y = inputs["img"].cuda(), inputs["label"].cuda()

        pred = self.model(x)

        loss = self.loss_fn(pred, y)

        if accuracy:
            pred_ = pred.argmax(dim=-1)

            accuracy = (pred_ == y).sum() / y.shape[0]

        return loss, accuracy


    def search(self, dl, support_img):
        l = []

        for batch in dl:
            logits = self.model(batch["img"])

            l += list(zip(batch["id"], logits))

        l = sorted(l, key=lambda tup: tup[1])
        
        return list(zip(*l))[0]