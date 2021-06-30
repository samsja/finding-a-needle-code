import pandas as pd
import sklearn.metrics as metrics
import torch

import torchvision

import numpy as np
from tqdm.autonotebook import tqdm
from src.few_shot_learning.standard_net import StandardNet, StandardNetAdaptater
from src.utils_search import (
    prepare_dataset,
    train_and_search,
    EntropyAdaptater,
    RandomAdaptater,
)

import pickle

from src.few_shot_learning.datasets import TrafficSignDataset
from src.few_shot_learning.utils_train import TrainerFewShot

from src.few_shot_learning import FewShotSampler2
from torchvision.models import resnet18

from src.few_shot_learning.datasets import FewShotDataSet
from collections import namedtuple


def get_transform():
    transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize(145),
            torchvision.transforms.RandomCrop(128),
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


FewShotParam = namedtuple(
    "FewShotParam",
    ["number_of_batch", "episodes", "sample_per_class", "classes_per_ep", "queries"],
)


few_shot_param = FewShotParam(5, 1, 1, 50, 8)


def init_few_shot_dataset(train_dataset, class_to_search_on, num_workers=5):

    train_few_shot_dataset = TrafficSignDataset(
        train_dataset.data,
        train_dataset.labels_str,
        train_dataset.transform,
        root_dir="",
        exclude_class=class_to_search_on,
    )

    few_shot_sampler = FewShotSampler2(
        train_few_shot_dataset,
        few_shot_param.number_of_batch,
        few_shot_param.episodes,
        few_shot_param.sample_per_class,
        few_shot_param.classes_per_ep,
        few_shot_param.queries,
    )

    few_shot_taskloader = torch.utils.data.DataLoader(
        train_few_shot_dataset, batch_sampler=few_shot_sampler, num_workers=num_workers
    )

    return few_shot_taskloader


def init_dataset(
    path_data, class_to_search_on, support_filenames, N=1, limit_search=None
):

    transform, transform_test = get_transform()

    with open("src/pickles/traineval_incl_partial.pkl", "rb") as f:
        train_eval = pickle.load(f)
        train_eval = [x for x in train_eval if "partial" not in x]

    with open("src/pickles/test_incl_partial.pkl", "rb") as f:
        test = pickle.load(f)
        test = [x for x in test if "partial" not in x]

    with open("src/pickles/class_list.pkl", "rb") as f:
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
            class_,
            support_index[class_],
            train_dataset,
            test_dataset,
            remove=True,
            limit_search=limit_search,
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

    search_model.load_state_dict(
        torch.load("data/results/active_loop/relation_net_model.pkl")
    )
    search_adaptater_relation_net = RelationNetAdaptater(
        search_model, 1, 1, 1, 1, device
    )

    return search_adaptater_relation_net


class Searcher:
    def train_searcher(
        self, train_dataset: FewShotDataSet, class_to_search_on, num_workers
    ):

        raise NotImplementedError


class NoAdditionalSearcher(Searcher):
    def __init__(self, model_adapter):
        self.model_adapter = model_adapter

    def train_searcher(self, *args, **kwargs):
        pass


class RelationNetSearcher(Searcher):

    lr = 3e-4
    epochs = 1
    nb_eval = 1

    def __init__(self, device,class_to_search_on):

        self.class_to_search_on = class_to_search_on

        self.model = RelationNet(
            in_channels=3,
            out_channels=64,
            embedding_module=ResNetEmbeddingModule(
                pretrained_backbone=resnet18(pretrained=True)
            ),
            relation_module=BasicRelationModule(input_size=512, linear_size=512),
            device=device,
            debug=True,
            merge_operator="mean",
        ).to(device)

        self.model_adapter = RelationNetAdaptater(self.model, 1, 1, 1, 1, device)

        self.train_model_adapter = RelationNetAdaptater(
            self.model,
            few_shot_param.episodes,
            few_shot_param.sample_per_class,
            few_shot_param.classes_per_ep,
            few_shot_param.queries,
            device,
        )

        self.device = device

        self.optim = torch.optim.Adam(
            self.model.parameters(), lr=RelationNetSearcher.lr
        )
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optim, step_size=1000, gamma=0.5
        )

        self.trainer = TrainerFewShot(
            self.train_model_adapter, device, checkpoint=False
        )

    def train_searcher(
        self, train_dataset: FewShotDataSet,  num_workers
    ):

        few_shot_task_loader = init_few_shot_dataset(
            train_dataset, self.class_to_search_on, num_workers
        )

        self.trainer.fit(
            RelationNetSearcher.epochs,
            RelationNetSearcher.nb_eval,
            self.optim,
            self.scheduler,
            few_shot_task_loader,
            few_shot_task_loader,
        )


def exp_active_loop(
    N,
    class_to_search_on,
    episodes,
    number_of_runs,
    top_to_select,
    epochs_step,
    nb_of_eval,
    lr,
    device,
    init_dataset,
    batch_size,
    model_adapter_search_param=None,
    search=True,
    callback=None,
    num_workers=4,
    retrain=True,
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
        "f_score": [],
        "train_size": [],
    }

    for run_id in tqdm(range(number_of_runs)):

        train_dataset, eval_dataset, test_dataset = init_dataset()

        train_loader = torch.utils.data.DataLoader(
            train_dataset, shuffle=True, num_workers=num_workers, batch_size=batch_size
        )

        val_loader = torch.utils.data.DataLoader(
            eval_dataset, shuffle=True, batch_size=batch_size, num_workers=num_workers
        )

        test_taskloader = torch.utils.data.DataLoader(
            test_dataset, num_workers=num_workers, batch_size=batch_size
        )

        for i in tqdm(range(episodes)):

            if retrain or i == 0:
                resnet_model = StandardNet(len(train_dataset.classes))
                resnet_model = resnet_model.to(device)
                resnet_adapt = StandardNetAdaptater(resnet_model, device)
                trainer = TrainerFewShot(resnet_adapt, device, checkpoint=True)

                optim_resnet = torch.optim.Adam(resnet_model.parameters(), lr=lr)

                scheduler_resnet = torch.optim.lr_scheduler.StepLR(
                    optim_resnet, step_size=100000, gamma=0.9
                )

            if model_adapter_search_param in [None, "StandardNet"]:
                model_adapter_search = resnet_adapt
                search_adaptater = NoAdditionalSearcher(model_adapter_search)

            elif model_adapter_search_param == "RelationNet":
                search_adaptater = RelationNetSearcher(device,class_to_search_on)

            elif model_adapter_search_param == "RelationNetFull":
                search_adaptater = RelationNetSearcher(device,[])

            elif model_adapter_search_param == "Entropy":
                model_adapter_search = EntropyAdaptater(resnet_model, device)
                search_adaptater = NoAdditionalSearcher(model_adapter_search)

            elif model_adapter_search_param == "Random":
                model_adapter_search = RandomAdaptater(resnet_model, device)
                search_adaptater = NoAdditionalSearcher(model_adapter_search)


            search_adaptater.train_searcher(
                train_dataset, num_workers
            )

            train_and_search(
                class_to_search_on,
                epochs_step[i],
                train_loader,
                val_loader,
                test_taskloader,
                trainer,
                optim_resnet,
                scheduler_resnet,
                search_adaptater.model_adapter,
                top_to_select=top_to_select,
                treshold=1,
                checkpoint=True,
                nb_of_eval=nb_of_eval[i],
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

            for class_ in class_to_search_on:
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

                scores["train_size"].append(
                    train_dataset.get_index_in_class(class_).shape[0]
                )

    if callback is not None:
        callback(resnet_model, train_dataset, eval_dataset, test_dataset)

    scores_df = pd.DataFrame(scores)

    return scores_df
