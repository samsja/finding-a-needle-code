import torch
import torch.nn.functional as F

from proto_net import ProtoNet
from few_shot_learning import FewShotSampler
from traffic_signs import TrafficSignDataset


def get_dist(center, features):
    """
        Args:
            centers : tensor. (512)
            features : tensor. (B, 512)

        Return:
            d : tensor. (B), distance between features and center

    """
    
    d = ((center-features)**2).sum(dim=-1)
    d = torch.sqrt(d)
    return d


def preprocess_batch(batch, device, NB, Q, K):
    x, _ = batch # (NB*(K+Q), 3, 224, 224) exepcted 
    x = x.view(NB, K+Q, 3, x.shape[-2], x.shape[-1])

    S_ = x[:, :K] # (NB, K, 3, 224, 224)
    Q_ = x[:, K:] # (NB, Q, 3, 224, 224)

    S_ = S_.contiguous().view(NB*K, 3, 224, 224)
    Q_ = Q_.contiguous.view(NB*Q, 3, 224, 224)

    return S_.to(device), Q_.to(device)



def loss_fn(centers, features):
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
        for f_ in q:
            dists = get_dist(centers, f_)
            
            tmp = dists[dist != dists[j]]
            tmp = torch.exp(-tmp).sum()

            mini_dist += dists[j]
            maxim_dist += torch.log(tmp)
    
    return (maxim_dist + mini_dist) / ( features.shape[0] * features.shape[1])


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

root_dir = "/workspaces/thesis_data_search/data/patches"

ds = TrafficSignDataset(root_dir, [])

K = 5 # Number of support vectors per class
Q = 5 # Number of query vectors per class
NB = 30 # Number of classes per episode

few_shot_sampler = FewShotSampler(
    ds,
    number_of_batch=100,
    episodes=1,
    sample_per_class=K,
    classes_per_ep=NB,
    queries=Q,
)

train_dl = torch.utils.data.DataLoader(
    ds, batch_sampler=few_shot_sampler, num_workers=1
)

val_dl = torch.utils.data.DataLoader(
    ds, batch_sampler=few_shot_sampler, num_workers=1
)

EPOCHS = range(100)

model = ProtoNet(3)
model = model.to(device)

optim = torch.optim.Adam(model.parameters, lr=0.001)

for epoch in EPOCHS:
    model.train()

    for batch in train_dl:
        S_, Q_ = preprocess_batch(batch, device, NB, Q, K) # (NB*K, 3, 224, 224), (NB*Q, 3, 224, 224)

        optim.zero_grad()

        C = model(S_) # (NB*K, 512)
        C = C.view(NB, K, 512)
        C = C.mean(dim=1) # (NB, 512)

        f = model(Q_) # (NB*Q, 512)
        f = f.view(NB, Q, 512)

        loss = loss_fn(C, f)
        loss.backward()

        optim.step()

    model.eval()

    total, correct = 0, 0

    for batch in val_dl:
        with torch.no_grad():
            S_, Q_ = preprocess_batch(batch, device, 5, 5, 5) # (NB*K, 3, 224, 224), (NB*Q, 3, 224, 224)

            C = model(S_)
            C = C.view(5, 5, 512)
            C = C.mean(dim=1) # (NB, 512)

            features = model(Q_)
            features = features.view(5, 5, 512)
            
            features = torch.swapaxes(features, 0,1)

            for i, queries in enumerate(features):
                center = C[i] # (512,)
                dist = get_dist(center, queries) # (NB)
                pred = F.softmax(-dist, dim=-1).argmax()

                correct += i == pred
                total += 1
                

    print("Accuracy: ", correct/total)

                
                



                




            

        




