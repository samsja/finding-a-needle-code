import torchvision
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import os
import torchvision.transforms.functional as TF
from tqdm import tqdm

from  few_shot_learning import FewShotDataSet, FewShotSampler
from  few_shot_learning.datasets import TrafficSignDataset,get_file_name_from_folder
from  few_shot_learning.utils_train import TrainerFewShot,RotationTransform,get_n_trainable_param
from few_shot_learning.relation_net import get_relation_net_adaptater

import argparse

##############
# Parameters #
##############

if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    parser = argparse.ArgumentParser()
    parser.add_argument("-n", default=5, type=int)
    parser.add_argument("-k", default=5, type=int)
    parser.add_argument("-q", default=1, type=int)
    parser.add_argument("--nb_ep", default=1, type=int)
    parser.add_argument("--nb_epochs", default=10, type=int)
    parser.add_argument("--nb_eval", default=10, type=int)
    parser.add_argument("-lr", default=1e-4, type=float)
    parser.add_argument("--nb_of_batch", default=10, type=int)
    parser.add_argument("--step_size", default=100, type=int)
    parser.add_argument("--model",default="RelationNet",type=str)

    args = parser.parse_args()

    nb_ep = args.nb_ep
    k = args.k
    q = args.q
    n = args.n
    number_of_batch = args.nb_of_batch
    step_size = args.step_size

    # ## load data

    class TrafficSignDatasetWrapper(TrafficSignDataset):

        def __init__(self,*args,**kwargs):
            super(TrafficSignDatasetWrapper,self).__init__(*args,**kwargs)

        def __getitem__(self,idx):
            data = super(TrafficSignDatasetWrapper,self).__getitem__(idx)
            return data["img"],data["label"]

    transform = torchvision.transforms.Compose([
                                            torchvision.transforms.Resize(256),
                                            torchvision.transforms.RandomCrop(224),
                                            torchvision.transforms.ToTensor(),
                                            torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                 std=[0.229, 0.224, 0.225])
                                           ])

    list_file_name_train = get_file_name_from_folder(root_dir="/staging/thesis_data_search/data/patches",exclude_class = torch.arange(201,313))
    list_file_name_val = get_file_name_from_folder(root_dir="/staging/thesis_data_search/data/patches",exclude_class = torch.arange(200))


    bg_dataset = TrafficSignDatasetWrapper(
        list_file_name_train,transform = transform
    )

    eval_dataset = TrafficSignDatasetWrapper(
        list_file_name_val,transform = transform
    )
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

    if args.model == "RelationNet":
        model_adaptater,model = get_relation_net_adaptater(nb_ep,n,k,q,device)
    else :
        raise NotImplementedError

    lr = args.lr

    optim = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=step_size, gamma=0.1)

   
    trainer = TrainerFewShot(model_adaptater, device)

    epochs = args.nb_epochs
    nb_eval = args.nb_eval

    trainer.fit(epochs, nb_eval, optim, scheduler, bg_taskloader, eval_taskloader)

    print(min(trainer.list_loss_eval), min(trainer.list_loss), max(trainer.accuracy_eval))

    few_shot_accuracy_sampler = FewShotSampler(
        bg_dataset,
        number_of_batch=1,
        episodes=nb_ep,
        sample_per_class=n,
        classes_per_ep=k,
        queries=q,
    )

    accuracy_taskloader = torch.utils.data.DataLoader(
        bg_dataset, batch_sampler=few_shot_accuracy_sampler, num_workers=10
    )

    print(trainer.accuracy(accuracy_taskloader))
