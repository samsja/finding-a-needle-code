import pandas as pd
import sklearn.metrics as metrics
import torch

import torchvision

import numpy as np
from tqdm.autonotebook import tqdm
from src.few_shot_learning.standard_net import StandardNet, StandardNetAdaptater
from src.utils_search import prepare_dataset, train_and_search, EntropyAdaptater

import pickle

from src.few_shot_learning.datasets import TrafficSignDataset
from src.few_shot_learning.utils_train import TrainerFewShot

from src.few_shot_learning import FewShotSampler2
from torchvision.models import resnet18

def get_transform():
    transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize(145),
            torchvision.transforms.RandomCrop(128)  ,
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),
        ]
    )

    transform_test = torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize((128, 128)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),
        ]
    )

    return transform, transform_test


def init_few_shot_dataset(train_dataset, class_to_search_on):

    train_few_shot_dataset = TrafficSignDataset(
        train_dataset.data,
        train_dataset.labels_str,
        train_dataset.transform,
        root_dir="",
        exclude_class=class_to_search_on,
    )

    few_shot_sampler = FewShotSampler2(
        train_few_shot_dataset,
        number_of_batch=5,
        episodes=1,
        sample_per_class=5,
        classes_per_ep=50,
        queries=8,
    )

    few_shot_taskloader = torch.utils.data.DataLoader(
        train_few_shot_dataset, batch_sampler=few_shot_sampler, num_workers=5
    )

    return few_shot_taskloader


def init_dataset(path_data, class_to_search_on, support_filenames, N=1,limit_search=None):

   
    transform, transform_test = get_transform()

    with open('src/pickles/traineval_incl_partial.pkl', 'rb') as f:
        train_eval = pickle.load(f)
        train_eval = [x for x in train_eval if "partial" not in x]

    with open('src/pickles/test_incl_partial.pkl', 'rb') as f:
        test = pickle.load(f)
        test = [x for x in test if "partial" not in x]

    with open('src/pickles/class_list.pkl', 'rb') as f:
        label_list = pickle.load(f)
        train_, val_ = [], []
    

    for c in label_list:
        fns = [x for x in test if c + "/" in x]
        ratio = int(len(fns) * 0.9) - 1
        train_ += fns[:ratio]
        val_ += fns[ratio:]

    train_dataset = TrafficSignDataset(
        train_, label_list, transform=transform, root_dir=path_data
    )

    eval_dataset = TrafficSignDataset(
        val_, label_list, transform=transform_test, root_dir=path_data
    )

    test_dataset = TrafficSignDataset(
        train_eval, transform=transform_test, root_dir=path_data, label_list=label_list
    )

    support_index = {}
    for class_ in class_to_search_on:

        if class_.item() not in support_filenames.keys():
            support_index[class_] = [
                train_dataset.data[train_dataset.get_index_in_class(class_)[i]]
                for i in range(N)
            ]
        else:
            support_index[class_.item()] = support_filenames[class_.item()]

    for class_ in support_index.keys():
 
        prepare_dataset(
            class_, support_index[class_], train_dataset, test_dataset, remove=True,limit_search=limit_search,
        )
    return train_dataset, eval_dataset, test_dataset


from src.few_shot_learning import RelationNet, RelationNetAdaptater
from src.few_shot_learning.relation_net import (
    BasicRelationModule,
    ResNetEmbeddingModule,
)
def get_relation_net(device):
    search_model = RelationNet(
    in_channels=3,
    out_channels=64,
    embedding_module=ResNetEmbeddingModule(
        pretrained_backbone=resnet18(pretrained=True)
    ),
    # embedding_module = ResNetEmbeddingModule(pretrained_backbone=resnet_model),
    relation_module=BasicRelationModule(input_size=512, linear_size=512),
    device=device,
    debug=True,
    merge_operator="mean",
    ).to(device)

    search_model.load_state_dict(torch.load("data/results/active_loop/relation_net_model.pkl"))
    search_adaptater_relation_net = RelationNetAdaptater(search_model,1,1,1,1,device)

    return search_adaptater_relation_net 



def print_param(f):
    def f2(*args,**kwargs):
        print(args)
        print(kwargs)
        return f(*args,**kwargs)
    return f2


#@print_param
def exp_active_loop(
    N,
    mask,
    episodes,
    number_of_runs,
    top_to_select,
    epochs_step,
    lr,
    device,
    init_dataset,
    batch_size,
    model_adapter_search=None,
    search=True,
    nb_of_eval=1, 
):

    scores = {
        "class": [],
        "iteration": [],
        "precision": [],
        "recall": [],
        "run_id": [],
        "acc": [],
        "TP": [],
        "FN": [],
        "FP": [],
        "f_score":[],
        "train_size": [],
    }

    for run_id in tqdm(range(number_of_runs)):

        train_dataset, eval_dataset, test_dataset = init_dataset()

        train_loader = torch.utils.data.DataLoader(
            train_dataset, shuffle=True, num_workers=4, batch_size=batch_size
        )

        val_loader = torch.utils.data.DataLoader(
            eval_dataset, shuffle=True, batch_size=batch_size, num_workers=4
        )

        test_taskloader = torch.utils.data.DataLoader(
            test_dataset, num_workers=10, batch_size=batch_size
        )

        for i in tqdm(range(episodes)):

            resnet_model = StandardNet(len(train_dataset.classes))
            resnet_model = resnet_model.to(device)
            resnet_adapt = StandardNetAdaptater(resnet_model, device)
            trainer = TrainerFewShot(resnet_adapt, device, checkpoint=True)

            optim_resnet = torch.optim.Adam(resnet_model.parameters(), lr=lr)

            scheduler_resnet = torch.optim.lr_scheduler.StepLR(
                optim_resnet, step_size=100, gamma=0.9
            )

            if model_adapter_search is None:
                model_adapter_search = resnet_adapt
  
            elif model_adapter_search == "StandardNet":
                model_adapter_search = resnet_adapt

            elif model_adapter_search == "RelationNet":
                model_adapter_search = get_relation_net(device)
            
            elif model_adapter_search == "Entropy":
                model_adapter_search = EntropyAdaptater(resnet_model,device)

            train_and_search(
                mask,
                epochs_step[i],
                train_loader,
                val_loader,
                test_taskloader,
                trainer,
                optim_resnet,
                scheduler_resnet,
                model_adapter_search,
                top_to_select=top_to_select,
                treshold=1,
                checkpoint=True,
                nb_of_eval=nb_of_eval,
                search=search,
            )

            outputs, true_labels = trainer.get_all_outputs(val_loader, silent=True)

            precision = torch.Tensor(
                metrics.precision_score(
                    true_labels.to("cpu"),
                    outputs.to("cpu"),
                    average=None,
                    zero_division=0,
                )
            )
            recall = torch.Tensor(
                metrics.recall_score(
                    true_labels.to("cpu"),
                    outputs.to("cpu"),
                    average=None,
                    zero_division=0,
                )
            )

            f_score = torch.Tensor(
                metrics.f1_score(
                    true_labels.to("cpu"),
                    outputs.to("cpu"),
                    average=None,
                    zero_division=0,
                )
            )
            accuracy = metrics.accuracy_score(true_labels.to("cpu"), outputs.to("cpu"))

            cf_matrix = torch.Tensor(
                metrics.confusion_matrix(true_labels.to("cpu"), outputs.to("cpu"))
            )

            FP = cf_matrix.sum(axis=0) - np.diag(cf_matrix)
            FN = cf_matrix.sum(axis=1) - np.diag(cf_matrix)
            TP = np.diag(cf_matrix)


            for class_ in mask:
                class_ = class_.item()
                scores["class"].append(class_)
                scores["precision"].append(precision[class_].item())
                scores["recall"].append(recall[class_].item())
                scores["TP"].append(TP[class_].item())
                scores["FN"].append(FN[class_].item())
                scores["FP"].append(FP[class_].item())
                scores["f_score"].append(f_score[class_].item()) 
                scores["iteration"].append(i)
                scores["run_id"].append(run_id)
                scores["acc"].append(accuracy)

                scores["train_size"].append(train_dataset.get_index_in_class(class_).shape[0]) 
                

    scores_df = pd.DataFrame(scores)


    return scores_df


