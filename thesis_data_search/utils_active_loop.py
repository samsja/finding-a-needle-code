import pandas as pd
import sklearn.metrics as metrics
import torch


import numpy as np
from tqdm.autonotebook import tqdm
from thesis_data_search.few_shot_learning.standard_net import StandardNet, StandardNetAdaptater
from thesis_data_search.utils_search import (
    train_and_search,
    EntropyAdaptater,
    RandomAdaptater,
)


import pickle

from thesis_data_search.few_shot_learning.utils_train import TrainerFewShot

from thesis_data_search.datasource import init_few_shot_dataset, few_shot_param, FewShotParam


from thesis_data_search.searcher.searcher import (
    RelationNetSearcher,
    ProtoNetSearcher,
    NoAdditionalSearcher,
    StandardNetSearcher,
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
                search_adaptater = RelationNetSearcher(
                    device, few_shot_param, class_to_search_on
                )

            elif model_adapter_search_param == "RelationNetFull":
                search_adaptater = RelationNetSearcher(device, few_shot_param, [])

            elif model_adapter_search_param == "ProtoNet":
                search_adaptater = ProtoNetSearcher(
                    device, few_shot_param, class_to_search_on
                )

            elif model_adapter_search_param == "ProtoNetFull":
                search_adaptater = ProtoNetSearcher(device, few_shot_param, [])

            elif model_adapter_search_param == "Entropy":
                model_adapter_search = EntropyAdaptater(resnet_model, device)
                search_adaptater = NoAdditionalSearcher(model_adapter_search)

            elif model_adapter_search_param == "Random":
                model_adapter_search = RandomAdaptater(resnet_model, device)
                search_adaptater = NoAdditionalSearcher(model_adapter_search)

            search_adaptater.train_searcher(train_dataset, eval_dataset, num_workers)

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
