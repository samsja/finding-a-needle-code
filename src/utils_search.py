import torch
import matplotlib.pyplot as plt
import pandas as pd
from tqdm.autonotebook import tqdm

from src.few_shot_learning.relation_net import (
    RelationNet,
    RelationNetAdaptater,
    get_relation_net_adaptater,
    get_features_for_one_class,
)

from src.few_shot_learning.standard_net import StandardNet, StandardNetAdaptater

from src.utils_plot import img_from_tensor, imshow, plot_list

from src.searcher.searcher import (
    RelationNetSearcher,
    ProtoNetSearcher,
    NoAdditionalSearcher,
    StandardNetSearcher,
)

from src.datasource import few_shot_param

def count_in_top(top, class_, test_dataset):
    top_labels = [test_dataset[idx]["label"].item() for idx in top]
    return top_labels.count(class_)


def plot_search(
    n,
    top,
    relation,
    test_dataset,
    figsize=(7, 7),
    img_from_tensor=img_from_tensor,
    ncols=4,
):
    top_n = torch.stack([test_dataset[i.item()]["img"] for i in top[0:n]])
    title = ["{:.2f}".format(r.item()) for r in relation[0:n]]

    plot_list(
        top_n,
        title=title,
        figsize=figsize,
        img_from_tensor=img_from_tensor,
        ncols=ncols,
    )


def plot_image_to_find(class_to_search_for, test_dataset, relation, top, max_len=10):

    index_to_find = test_dataset.get_index_in_class(class_to_search_for)
    max_len = min(max_len, len(index_to_find) - 1)

    index_to_find = index_to_find[:max_len]

    top = top.squeeze(1)
    relation = torch.stack([relation[top == i] for i in index_to_find])

    title = ["{:.2f}".format(r.item()) for r in relation]

    target_images = torch.stack([test_dataset[i]["img"] for i in index_to_find])

    plot_list(target_images, title=title)


def search_rare_class(
    class_to_search_for,
    idx_support,
    model,
    train_dataset,
    test_dataset,
    batch_size,
    test_taskloader,
    device,
    model_adaptater,
    plot=False,
    tqdm_silent=False,
    max_len=10,
):
    support_img = torch.stack([train_dataset[idx]["img"] for idx in idx_support])

    top, relation = model_adaptater.search_tensor(
        test_taskloader, support_img, class_to_search_for, tqdm_silent=tqdm_silent
    )

    if plot:
        imshow(support_img[0])

    t20 = count_in_top(top[:20], class_to_search_for, test_dataset)
    t100 = count_in_top(top[:100], class_to_search_for, test_dataset)
    t5 = count_in_top(top[:5], class_to_search_for, test_dataset)
    t1000 = count_in_top(top[:1000], class_to_search_for, test_dataset)

    if plot:
        plot_search(
            50,
            top,
            relation,
            test_dataset,
            figsize=(15, 15),
            ncols=6,
        )

        plot_image_to_find(
            class_to_search_for, test_dataset, relation, top, max_len=max_len
        )

    return (
        len(test_dataset.get_index_in_class(class_to_search_for)),
        t5,
        t20,
        t100,
        t1000,
        top,
        relation,
    )


def move_found_images(datapoint_to_add, train_dataset, test_dataset):
    len_train, len_test = len(train_dataset), len(test_dataset)

    for class_ in datapoint_to_add.keys():
        for datapoint in datapoint_to_add[class_]:
            train_dataset.add_datapoint(
                test_dataset.data[datapoint], test_dataset.labels_str[class_]
            )

    for class_ in datapoint_to_add.keys():
        test_dataset.remove_datapoints(datapoint_to_add[class_])

    train_dataset.update_classes_indexes()
    test_dataset.update_classes_indexes()

    # assert(len(train_dataset) + len(test_dataset)  == len_train + len_test)


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


def plot_score_one_class(score, class_, scores_df):
    plt.plot(list(scores_df[scores_df["class"] == class_][score]), label=f"{class_}")


def plot_mean(score, scores_df):
    plt.plot(
        list(scores_df[["iteration", score]].groupby(["iteration"]).mean()[score]),
        label="mean",
        linewidth=4,
    )


def plot_score(score, scores_df):
    for class_ in scores_df["class"].unique():
        plot_score_one_class(score, class_, scores_df)

    plot_mean(score, scores_df)
    plt.title(f"{score}")
    plt.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.05),
        fancybox=True,
        shadow=True,
        ncol=5,
    )


def plot_score_model(score, model, scores_df):
    plot_score(score, scores_df[scores_df["model"] == model])


def plot_all_model_mean(score, scores_df):
    for model in scores_df["model"].unique():
        df = scores_df[scores_df["model"] == model]
        plt.plot(
            list(df[["iteration", score]].groupby(["iteration"]).mean()[score]),
            label=model,
        )
    plt.title(f"{score}")
    plt.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.05),
        fancybox=True,
        shadow=True,
        ncol=5,
    )


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
    top_to_select : list,
    device,
    init_dataset,
    batch_size,
    model_adapter_search_param=None,
    callback=None,
    num_workers=4,
):

    scores = {
        "class": [],
        "run_id": [],
        "data_available" : [],
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
            search_adaptater = StandardNetSearcher(device, len(train_dataset.classes))

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

        ## search

        for class_ in class_to_search_on:



            top, relation = search_and_get_top(
                search_adaptater.model_adapter,
                class_,
                test_taskloader,
                train_loader.dataset,
            )
            #  plot_search(
                #      50,
                #      top,
                #      relation,
                #      test_dataset,
                #      figsize=(15, 15),
                #      ncols=6,
                #  )
            #

            class_ = class_.item()
            data_available = len(test_dataset.get_index_in_class(class_)) 
            scores["class"].append(class_)
            scores["run_id"].append(run_id)
            scores["data_available"].append(data_available)


            for topX in top_to_select:
                count = count_in_top(top[:topX], class_, test_dataset)
                scores[f"t{topX}"].append(count)
                scores[f"s{topX}"].append(count/min(topX,data_available))

        if callback is not None:
            callback(train_dataset, eval_dataset, test_dataset)

    scores_df = pd.DataFrame(scores)

    return scores_df
