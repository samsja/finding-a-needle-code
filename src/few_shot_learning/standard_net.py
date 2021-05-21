import torch.nn as nn
import torch.nn.functional as F
import torch

from torchvision.models import resnet18
from .utils_train import ModuleAdaptater

from tqdm.autonotebook import tqdm

class StandardNet(nn.Module):
    """
    Implementation of a standard ResNet classifier in pytorch

    # Arguments
        classes: int. Number of output nodes
    """

    def __init__(self, classes, pretrained=True):
        super().__init__()

        self.resnet = resnet18(pretrained=pretrained)

        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, classes)

    def forward(self, x):

        return self.resnet(x)

    def freeze_firsts_layer(self):

        for p in self.parameters():
            p.requires_grad = False

        for layer in [self.resnet.layer4, self.resnet.fc]:
            for p in layer.parameters():
                p.requires_grad = True

    def freeze_mlp(self):

        for p in self.parameters():
            p.requires_grad = False

        for p in self.resnet.fc.parameters():
            p.requires_grad = True



    def unfreeze(self):

        for p in self.parameters():
            p.requires_grad = True


class StandardNetAdaptater(ModuleAdaptater):
    """
    Standard net module adaptater
    """

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        nb_ep: int = None,
        n: int = None,
        k: int = None,
        q: int = None,
    ):
        super().__init__(model)

        self.device = device
        self.model = model
        self.loss_fn = nn.CrossEntropyLoss()

    def get_loss_and_accuracy(self, inputs, labels, accuracy=False):
        outputs = self.model(inputs)
        loss = self.loss_fn(outputs, labels.long())

        if accuracy:
            _, preds = torch.max(outputs, 1)
            #print((preds==25).float().sum())
            return loss, (preds == labels).float().mean()

        else:
            return loss, None

    def search(self, dl, support_img, rare_class_index):
        l = []

        for batch in dl:
            with torch.no_grad():
                logits = self.model(batch["img"].to(device))[:, rare_class_index].tolist()

            l += list(zip(batch["id"], logits))

        l = sorted(l, key=lambda tup: -tup[1])

        return list(zip(*l))[0]

    
    @torch.no_grad()
    def search_tensor(
        self,
        test_taskloader: torch.utils.data.DataLoader,
        support_set: torch.Tensor,
        rare_class_index: int,
        tqdm_silent = False,
    ):

        self.model.eval()

        relations = []
        index = []
        for idx, batch in enumerate(tqdm(test_taskloader,disable=tqdm_silent)):

            query_inputs = batch["img"].to(self.device)

            logits = F.softmax(self.model(batch["img"].to(self.device)),dim=1)[:, rare_class_index]
    
            relations.append(logits)
            index.append(batch["id"].long().to(self.device))

        index = torch.cat(index)
        relations = torch.cat(relations)

        relations, argsort = torch.sort(relations, descending=True)

        return index[argsort], relations


