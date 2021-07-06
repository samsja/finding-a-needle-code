import torchvision
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import pickle
import os
import torchvision.transforms.functional as TF
from tqdm import tqdm

from  few_shot_learning import FewShotDataSet, FewShotSampler2, RelationNet, ProtoNet, RelationNetAdaptater, ProtoNetAdaptater, StandardNet, StandardNetAdaptater
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
    parser.add_argument("-rare_class_index", type=int)

    args = parser.parse_args()

    nb_ep = args.nb_ep
    k = args.k
    q = args.q
    n = args.n
    number_of_batch = args.nb_of_batch
    step_size = args.step_size

    # ## load data


    transform = torchvision.transforms.Compose([
                                            torchvision.transforms.Resize(145),
                                            torchvision.transforms.RandomRotation(degrees=(-30, 30)),
                                            torchvision.transforms.RandomCrop(128),
                                            torchvision.transforms.ColorJitter(brightness=0.6, contrast=0.4, saturation=0.7, hue=0.05),
                                            torchvision.transforms.ToTensor(),
                                            torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                 std=[0.229, 0.224, 0.225]),
        
                                            torchvision.transforms.GaussianBlur(23, sigma=(0.1, 2.0)),

                                            torchvision.transforms.RandomErasing(p=0.1,scale=(0.02, 0.20), ratio=(0.1, 1.1),value="random"),

        
                                           ])


    transform_test= torchvision.transforms.Compose([
                                            torchvision.transforms.Resize((128,128)),
                                            torchvision.transforms.ToTensor(),
                                            torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                 std=[0.229, 0.224, 0.225])
                                           ])




    with open('traineval_incl_partial.pkl', 'rb') as f:
        train_eval = pickle.load(f)

    with open('test_incl_partial.pkl', 'rb') as f:
        test = pickle.load(f)

    with open('class_list.pkl', 'rb') as f:
        label_list = pickle.load(f)


    train_dataset = TrafficSignDataset(
        train_eval, label_list, transform = transform
    )


    eval_dataset = TrafficSignDataset(
        train_eval, label_list, transform = transform_test
    )


    test_dataset = TrafficSignDataset(
        test, label_list, transform = transform_test
    )

    

    few_shot_sampler = FewShotSampler2(
        train_dataset,
        number_of_batch=number_of_batch,
        episodes=nb_ep,
        sample_per_class=n,
        classes_per_ep=k,
        queries=q,
    )

    few_shot_sampler_val = FewShotSampler2(
        eval_dataset,
        number_of_batch=number_of_batch,
        episodes=nb_ep,
        sample_per_class=n,
        classes_per_ep=k,
        queries=q,
    )

    # In[15]:

    train_taskloader = torch.utils.data.DataLoader(
        train_dataset, batch_sampler=few_shot_sampler, num_workers=10
    )

    eval_taskloader = torch.utils.data.DataLoader(
        eval_dataset, batch_sampler=few_shot_sampler_val, num_workers=10
    )

    test_taskloader = torch.utils.data.DataLoader(
        test_dataset, batch_size=128, num_workers=10
    )

    # In[26]:

    if args.model == "RelationNet":
        model = RelationNet()
        model_adaptater = RelationNetAdaptater(model, nb_ep,n,k,q,device)
    if args.model == "ProtoNet" :
        model = ProtoNet(3)
        model_adaptater = ProtoNetAdaptater(model)
    else:
        model = StandardNet(len(label_list))
        model_adaptater = StandardNetAdaptater(model)

    lr = args.lr

    rare_class_index = args.rare_class_index


    for _ in range(10):
        optim = torch.optim.Adam(model_adaptater.model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=step_size, gamma=0.1)
    
        trainer = TrainerFewShot(model_adaptater, device)

        epochs = args.nb_epochs
        nb_eval = args.nb_eval

        trainer.fit(epochs, nb_eval, optim, scheduler, train_taskloader, eval_taskloader)

        # Perform search

        model_adaptater.model = trainer.model_checkpoint

        support_img = train_dataset.get_support(5, rare_class_index)
        index_list = model_adaptater.search(test_taskloader, support_img, rare_class_index)

        order_ = []

        for i, idx in enumerate(index_list:
            fn = test_dataset.data[idx]
            c = test_dataset.labels[idx]

            if c == rare_class_index:
                order_.append(i)

            train_dataset.add_datapoint(fn, c)
            
        test_dataset.remove_datapoints(index_list)

        print("Ordering:", order_)

        train_dataset.update_indices()
        test_dataset.update_indices()









