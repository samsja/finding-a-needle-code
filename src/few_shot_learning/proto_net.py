import torch.nn as nn
import torch.nn.functional as F
import torch

from torchvision.models import resnet18
from .utils_train import ModuleAdaptater


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





class ProtoNetAdaptater(ModuleAdaptater):
    """
    Proto net module adaptater
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
        super(ProtoNetAdaptater, self).__init__(model)
        self.nb_ep = nb_ep
        self.n = n
        self.k = k
        self.q = q
        self.device = device
        


    def preprocess_batch(self, batch, device, NB, Q, K):
        x, _ = batch # (NB*(K+Q), 3, 224, 224) exepcted 
        x = x.view(NB, K+Q, 3, x.shape[-2], x.shape[-1])

        S_ = x[:, :K] # (NB, K, 3, 224, 224)
        Q_ = x[:, K:] # (NB, Q, 3, 224, 224)

        S_ = S_.contiguous().view(NB*K, 3, 224, 224)
        Q_ = Q_.contiguous.view(NB*Q, 3, 224, 224)

        return S_.to(device), Q_.to(device)


    def get_dist(self, center, features):
    """
        Args:
            centers : tensor. (512)
            features : tensor. (_, 512)

        Return:
             : tensor.  distance between features and center

    """
    return ((center-features)**2).sum(dim=-1)


    def loss_fn(self, centers, features):
    """
    Args:
        centers : tensor. (NB, 512)
        features : tensor. (NB, Q, 512)

    Return:
        loss : tensor. Total loss for batch

    """

    # Calculate loss
    mini_dist = 0
    maxim_dist = 0

    for j, q in enumerate(features):
        ind = list(range(centers.shape[0]))
        del ind[j]
        
        for f_ in q:
            dists = get_dist(centers, f_)
            
            mini_dist += dists[j]
            
            tmp = torch.clamp(dists[ind], 0, 40)
            tmp = torch.exp(-tmp).sum()
            
            maxim_dist += torch.log(tmp)
            
    return (maxim_dist + mini_dist) / (features.shape[0] * features.shape[1])

    def get_loss_and_accuracy(self, inputs, labels, accuracy=False):

        S_, Q_ = self.preprocess_batch(inputs["img"], 
                                        self.device, 
                                        self.n, 
                                        self.q, 
                                        self.k) 

        # S_, Q_ = (NB*K, 3, 224, 224), (NB*Q, 3, 224, 224)

        C = self.model(S_) # (NB*K, 512)
        C = C.view(self.n, self.k, 512)
        C = C.mean(dim=1) # (NB, 512)

        f = self.model(Q_) # (NB*Q, 512)
        f = f.view(self.n, self.q, 512)

        loss = self.loss_fn(C, f)

        features = torch.swapaxes(f, 0,1)

        if accuracy:
            total, correct = 0, 0

            for i, cent in enumerate(C):
                dist = self.get_dist(features, cent)
                pred = F.softmax(-dist, dim=-1).argmax(dim=-1)

                correct += (pred == i).sum()
                total += pred.shape[0]
        
            accuracy = correct/total

        return loss, accuracy


    def search(self, dl, support_vector):
        l = []

        for batch in dl:
            output = self.model(batch["img"])

            dists = self.get_dist(support_vector, output).tolist()


            l += (batch["id"].tolist(), dists)

        l.sort()
        return l