import torchvision
import matplotlib.pyplot as plt
import torch

from torchvision.datasets.utils import list_files,list_dir
from os.path import join
import os
from typing import List,Tuple,Any




from few_shot_learning import FewShotDataSet, FewShotSampler,Omniglot

bg_dataset = Omniglot(
    root="./data", download=True, transform=torchvision.transforms.ToTensor(),background=True
)

eval_dataset = Omniglot(
    root="./data", download=True, transform=torchvision.transforms.ToTensor(),background=False
)


few_shot_sampler = FewShotSampler(bg_dataset,batch_size=10,episodes=2,sample_per_class=3,classes_per_ep=4,queries=2)


bg_taskloader = torch.utils.data.DataLoader(
    bg_dataset,
    batch_sampler= few_shot_sampler,
    num_workers=1
)


nb_ep = few_shot_sampler.episodes
k = few_shot_sampler.classes_per_ep
n = few_shot_sampler.sample_per_class
q = few_shot_sampler.queries

epochs = range(10)

for epoch in epochs:
    for batch_idx,batch in enumerate(bg_taskloader):

        inputs,labels = batch

        inputs = inputs.view(nb_ep,k,n+q,1,105,105)
        labels = labels.view(nb_ep,k,n+q)

        support_inputs,support_labels = inputs[:,:,:n] , labels[:,:,:n] 
        queries_inputs, queries_labels = inputs[:,:,-q:] ,labels[:,:,-q:]

        """
        training phase
        """
