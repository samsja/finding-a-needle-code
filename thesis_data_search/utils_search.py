import torch
import pandas as pd
from tqdm.autonotebook import tqdm

from thesis_data_search.few_shot_learning.relation_net import (
    RelationNet,
    RelationNetAdaptater,
    get_relation_net_adaptater,
    get_features_for_one_class,
)

from thesis_data_search.few_shot_learning.standard_net import StandardNet, StandardNetAdaptater

from thesis_data_search.utils_plot import imshow

from thesis_data_search.searcher.searcher import (
    RelationNetSearcher,
    ProtoNetSearcher,
    NoAdditionalSearcher,
    StandardNetSearcher,
    FreezedStandardNetSearcher,
)

from thesis_data_search.datasource import few_shot_param

from thesis_data_search.utils_plot import (
    plot_search,
    plot_image_to_find,
)


def count_in_top(top, class_, test_dataset):
    top_labels = [test_dataset[idx]["label"].item() for idx in top]
    return top_labels.count(class_)


import copy


def move_found_images(datapoint_to_add, train_dataset, test_dataset):
    len_train, len_test = len(train_dataset), len(test_dataset)

    for class_ in datapoint_to_add.keys():
        for datapoint in datapoint_to_add[class_]:
            train_dataset.add_datapoint(
                test_dataset.data[datapoint], test_dataset.labels_str[class_]
            )

    idx_to_remove = []
    for class_ in datapoint_to_add.keys():
        for idx in datapoint_to_add[class_]:
            idx_to_remove.append(idx)

    test_dataset.remove_datapoints(idx_to_remove)

    train_dataset.update_classes_indexes()
    test_dataset.update_classes_indexes()


def train_and_search(
    mask,
    epochs,
    train_loader,
    val_loader,
    test_taskloader,
    trainer,
    optim_resnet,
    scheduler_resnet,
    model_adaptater_search,
    top_to_select=1,
    treshold=0.5,
    only_true_image=True,
    checkpoint=True,
    nb_of_eval=1,
    search=True,
):

    trainer.fit(
        epochs,
        nb_of_eval,
        optim_resnet,
        scheduler_resnet,
        train_loader,
        val_loader,
        silent=True,
    )
    if checkpoint:
        trainer.model_adaptater.model = trainer.model_checkpoint

    if search:
        class_to_rebalanced = mask

        for class_ in class_to_rebalanced:
            datapoint_to_add = found_new_images(
                model_adaptater_search,
                class_,
                test_taskloader,
                train_loader.dataset,
                top_to_select=top_to_select,
            )
            move_found_images(
                datapoint_to_add, train_loader.dataset, test_taskloader.dataset
            )


def search_and_get_top(
    model_adapter,
    class_,
    test_taskloader,
    train_dataset,
):
    support_set = torch.stack(
        [train_dataset[idx]["img"] for idx in train_dataset.get_index_in_class(class_)]
    )

    top, relation = model_adapter.search_tensor(
        test_taskloader,
        support_set,
        class_,
        tqdm_silent=True,
    )

    return top, relation


def found_new_images(
    model_adapter,
    class_,
    test_taskloader,
    train_dataset,
    top_to_select=50,
):

    top, _ = search_and_get_top(
        model_adapter,
        class_,
        test_taskloader,
        train_dataset,
    )

    topX = top[:top_to_select][:, 0]

    datapoint_to_add = {}

    for data in topX:

        class_data = test_taskloader.dataset[data]["label"].item()

        if class_data not in datapoint_to_add.keys():
            datapoint_to_add[class_data] = [data]
        else:
            datapoint_to_add[class_data].append(data)

    for class_data in datapoint_to_add.keys():
        datapoint_to_add[class_data] = torch.Tensor(datapoint_to_add[class_data]).long()

    return datapoint_to_add


class EntropyAdaptater:
    def __init__(self, model, device):
        self.model = model
        self.device = device

    @torch.no_grad()
    def search_tensor(
        self,
        test_taskloader: torch.utils.data.DataLoader,
        support_set: torch.Tensor,
        rare_class_index=None,
        tqdm_silent=False,
    ):

        self.model.eval()

        entropy = []
        index = []
        for idx, batch in enumerate(tqdm(test_taskloader, disable=tqdm_silent)):

            inputs = batch["img"].to(self.device)

            outputs = self.model(inputs).softmax(dim=1)

            batch_entropy = -(torch.log(outputs) * outputs).sum(dim=1)

            entropy.append(batch_entropy)
            index.append(batch["id"].long().to(self.device))

        index = torch.cat(index)
        entropy = torch.cat(entropy)

        entropy, argsort = torch.sort(entropy, descending=True)

        return index[argsort], entropy


class RandomAdaptater:
    def __init__(self, model, device):
        self.device = device
        self.model = model

    @torch.no_grad()
    def search_tensor(
        self,
        test_taskloader: torch.utils.data.DataLoader,
        support_set: torch.Tensor,
        rare_class_index=None,
        tqdm_silent=False,
    ):

        index = torch.randperm(len(test_taskloader.dataset)).long().unsqueeze(dim=1)
        value = torch.zeros(len(test_taskloader.dataset)).float()

        return index, value


def exp_searching(
    N,
    class_to_search_on,
    number_of_runs,
    top_to_select: list,
    device,
    init_dataset,
    batch_size,
    model_adapter_search_param=None,
    callback=None,
    num_workers=4,
    plot=False,
    debug=False,
):

    scores = {
        "class": [],
        "run_id": [],
        "data_available": [],
    }

    for top in top_to_select:
        scores[f"t{top}"] = []
        scores[f"s{top}"] = []

    for run_id in tqdm(range(number_of_runs)):

        train_dataset, eval_dataset, test_dataset = init_dataset()

        train_loader = torch.utils.data.DataLoader(
            train_dataset, shuffle=True, num_workers=num_workers, batch_size=batch_size
        )

        test_taskloader = torch.utils.data.DataLoader(
            test_dataset, num_workers=num_workers, batch_size=batch_size
        )

        if model_adapter_search_param in [None, "StandardNet"]:
            search_adaptater = StandardNetSearcher(
                device, len(train_dataset.classes), debug=debug
            )

        elif model_adapter_search_param == "RelationNet":
            search_adaptater = RelationNetSearcher(
                device, few_shot_param, class_to_search_on, debug=debug
            )

        elif model_adapter_search_param == "RelationNetFull":
            search_adaptater = RelationNetSearcher(
                device, few_shot_param, [], debug=debug
            )

        elif model_adapter_search_param == "ProtoNet":
            search_adaptater = ProtoNetSearcher(
                device, few_shot_param, class_to_search_on, debug=debug
            )

        elif model_adapter_search_param == "ProtoNetFull":
            search_adaptater = ProtoNetSearcher(device, few_shot_param, [], debug=debug)

        elif model_adapter_search_param == "FreezedStandardNet":
            search_adaptater = FreezedStandardNetSearcher(
                device,
                len(train_dataset.classes),
                class_to_search_on=class_to_search_on,
                debug=debug,
            )
        else:
            raise ValueError(f"{model_adapter_search_param} is incorect")

        search_adaptater.train_searcher(train_dataset, eval_dataset, num_workers)

        ## search

        for class_ in class_to_search_on:

            top, relation = search_and_get_top(
                search_adaptater.model_adapter,
                class_,
                test_taskloader,
                train_loader.dataset,
            )

            if plot:
                imshow(
                    train_dataset[train_dataset.get_index_in_class(class_)[0]]["img"]
                )
                plot_search(
                    50,
                    top,
                    relation,
                    test_dataset,
                    figsize=(9, 15),
                    ncols=6,
                )

                plot_image_to_find(
                    class_,
                    test_dataset,
                    relation,
                    top,
                    max_len=50,
                )

            class_ = class_.item()
            data_available = len(test_dataset.get_index_in_class(class_))
            scores["class"].append(class_)
            scores["run_id"].append(run_id)
            scores["data_available"].append(data_available)

            for topX in top_to_select:
                count = count_in_top(top[:topX], class_, test_dataset)
                scores[f"t{topX}"].append(count)
                scores[f"s{topX}"].append(count / min(topX, data_available))

        if callback is not None:
            callback(train_dataset, eval_dataset, test_dataset)

    scores_df = pd.DataFrame(scores)

    return scores_df
