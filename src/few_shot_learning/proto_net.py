import torch.nn as nn
import torch.nn.functional as F
import torch

from torchvision.models import resnet18
from .utils_train import ModuleAdaptater
from tqdm.auto import tqdm

class ProtoNet(torch.nn.Module):
    """
    Implementation of the ProtoNet model in pytorch

    # Arguments
        in_channels: int. number of input channels
        debug: bool. if true debug mode activate defaulf  False
    """

    def __init__(self, in_channels: int, debug: bool = False,pretrain=True):
        super().__init__()

        self.debug = debug

        resnet = list(resnet18(pretrained=pretrain).children())[:-1]
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
        super().__init__(model)
        self.nb_ep = nb_ep
        self.n = n
        self.k = k
        self.q = q
        self.device = device
        self.model = model

    def preprocess_batch(self, batch, device, N, Q, K):
        x = batch  # (K*(N+Q), 3, H, W) exepcted
        H, W = x.shape[-2], x.shape[-1]
        x = x.view(K, N + Q, 3, H, W)
        S_ = x[:, :N]  # (NB, K, 3, H, W)
        Q_ = x[:, N:]  # (NB, Q, 3, H, W)
        S_ = S_.contiguous().view(K * N, 3, H, W)
        Q_ = Q_.contiguous().view(K * Q, 3, H, W)
        return S_.to(device), Q_.to(device)

    def get_dist(self, center, features):
        """
        Args:
            centers : tensor. (512)
            features : tensor. (_, 512)

        Return:
            : tensor.  distance between features and center (_, 1)

        """

        return ((center - features) ** 2).sum(dim=-1)

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
                dists = self.get_dist(centers, f_)

                mini_dist += dists[j]

                tmp = torch.clamp(dists[ind], 0, 40)
                tmp = torch.exp(-tmp).sum()

                maxim_dist += torch.log(tmp)

        return (maxim_dist + mini_dist) / (features.shape[0] * features.shape[1])

    def get_loss_and_accuracy(self, inputs, labels, accuracy=False):

        S_, Q_ = self.preprocess_batch(inputs, self.device, self.n, self.q, self.k)

        # S_, Q_ = (K*N, 3, 224, 224), (K*Q, 3, 224, 224)

        C = self.model(S_)  # (K*N, 512)
        C = C.view(self.k, self.n, 512)
        C = C.mean(dim=1)  # (K 512)

        f = self.model(Q_)  # (K*q, 512)
        f = f.view(self.k, self.q, 512)

        loss = self.loss_fn(C, f)

        features = torch.swapaxes(f, 0, 1)

        if accuracy:
            total, correct = 0, 0

            for i, cent in enumerate(C):
                dist = self.get_dist(features, cent)
                pred = F.softmax(-dist, dim=-1).argmax(dim=-1)

                correct += (pred == i).sum()
                total += pred.shape[0]

            accuracy = correct / total

        return loss, accuracy

    def search(self, dl, support_img, rare_class_index):
        l = []

        with torch.no_grad():
            support_vector = self.model(support_img.cuda())  # (N, 512)

        support_vector = support_vector.mean(dim=0)  # (512)

        for batch in dl:
            with torch.no_grad():
                output = self.model(batch["img"].cuda())

            dists = self.get_dist(support_vector, output).squeeze(-1).tolist()

            l += list(zip(batch["id"], dists))

        l = sorted(l, key=lambda tup: tup[1])

        return list(zip(*l))[0]

    @torch.no_grad()
    def search_tensor(
        self,
        test_taskloader: torch.utils.data.DataLoader,
        support_set: torch.Tensor,
        rare_class_index: int,
        tqdm_silent=False,
    ):

        self.model.eval()

        dists = []
        index = []

        support_vector = self.model(support_set.to(self.device)).mean(dim=0)  # (N, 512)

        for idx, batch in enumerate(tqdm(test_taskloader, disable=tqdm_silent)):

            query_inputs = batch["img"].to(self.device)
            
            query_features = self.model(query_inputs.to(self.device))

            batch_dists = self.get_dist(support_vector, query_features).squeeze(-1)


            dists.append(batch_dists)
            index.append(batch["id"].long().to(self.device))

        index = torch.cat(index)
        dists = torch.cat(dists)

        dists, argsort = torch.sort(dists)

        return index[argsort], dists
