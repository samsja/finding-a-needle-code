import torchvision
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import os
import torchvision.transforms.functional as TF
from tqdm import tqdm
from few_shot_learning import (
    FewShotDataSet,
    FewShotSampler,
    Omniglot,
    RelationNet,
    RelationNetAdaptater,
)
from few_shot_learning import TrainerFewShot, RotationTransform

import argparse

##############
# Parameters #
##############

if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    parser = argparse.ArgumentParser()
    parser.add_argument("-n", default=5, type=int)
    parser.add_argument("-k", default=5, type=int)
    parser.add_argument("-q", default=15, type=int)
    parser.add_argument("--nb_ep", default=10, type=int)
    parser.add_argument("--nb_epochs", default=10, type=int)
    parser.add_argument("--nb_eval", default=20, type=int)
    parser.add_argument("-lr", default=1e-4, type=float)
    parser.add_argument("--nb_of_batch", default=10, type=int)
    parser.add_argument("--step_size", default=100, type=int)

    args = parser.parse_args()

    nb_ep = args.nb_ep
    k = args.k
    q = args.q
    n = args.n
    number_of_batch = args.nb_of_batch
    step_size = args.step_size

    transform = torchvision.transforms.Compose(
        [
            RotationTransform(angles=[90, 180, 270], fill=255),
            torchvision.transforms.Resize(28, interpolation=2),
            torchvision.transforms.ToTensor(),
        ]
    )

    transform_eval = torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize(28),
            torchvision.transforms.ToTensor(),
        ]
    )

    transform = torchvision.transforms.Compose(
        [
            # torchvision.transforms.RandomRotation(90,expand=True),
            # torchvision.transforms.RandomRotation(180,expand=True),
            # torchvision.transforms.RandomRotation(270,expand=True),
            torchvision.transforms.Resize(28),
            torchvision.transforms.ToTensor(),
        ]
    )

    bg_dataset = Omniglot(
        root="/staging/thesis_data_search/data",
        download=False,
        transform=transform,
        background=True,
    )

    eval_dataset = Omniglot(
        root="/staging/thesis_data_search/data",
        download=False,
        transform=transform_eval,
        background=False,
    )

    # ## Train

    # In[12]:

    few_shot_sampler = FewShotSampler(
        bg_dataset,
        number_of_batch=number_of_batch,
        episodes=nb_ep,
        sample_per_class=n,
        classes_per_ep=k,
        queries=q,
    )

    few_shot_sampler_val = FewShotSampler(
        eval_dataset,
        number_of_batch=number_of_batch,
        episodes=nb_ep,
        sample_per_class=n,
        classes_per_ep=k,
        queries=q,
    )

    # In[15]:

    bg_taskloader = torch.utils.data.DataLoader(
        bg_dataset, batch_sampler=few_shot_sampler, num_workers=10
    )

    eval_taskloader = torch.utils.data.DataLoader(
        eval_dataset, batch_sampler=few_shot_sampler_val, num_workers=10
    )

    # In[26]:

    loss_func = nn.MSELoss()
    model = RelationNet(in_channels=1, out_channels=64, device=device, debug=True)
    model.to(device)

    lr = args.lr

    optim = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=step_size, gamma=0.1)

    model_adaptater = RelationNetAdaptater(model, loss_func, nb_ep, n, k, q, device)
    trainer = TrainerFewShot(model_adaptater, device)

    epochs = args.nb_epochs
    nb_eval = args.nb_eval

    trainer.fit(epochs, nb_eval, optim, scheduler, bg_taskloader, eval_taskloader)

    min(trainer.list_loss_eval), min(trainer.list_loss), max(trainer.accuracy_eval)

    few_shot_accuracy_sampler = FewShotSampler(
        bg_dataset,
        number_of_batch=100,
        episodes=nb_ep,
        sample_per_class=n,
        classes_per_ep=k,
        queries=q,
    )

    accuracy_taskloader = torch.utils.data.DataLoader(
        bg_dataset, batch_sampler=few_shot_accuracy_sampler, num_workers=10
    )

    print(trainer.accuracy(accuracy_taskloader))
