import torch

from tqdm.autonotebook import tqdm

from src.few_shot_learning.relation_net import (
    RelationNet,
    RelationNetAdaptater,
    get_relation_net_adaptater,
    get_features_for_one_class,
)

from src.few_shot_learning.standard_net import StandardNet, StandardNetAdaptater

from src.utils_plot import img_from_tensor, imshow, plot_list


def prepare_dataset(
    class_to_search_for, idx_support, train_dataset, test_dataset, remove=False
):

    index_in_class = train_dataset.get_index_in_class(class_to_search_for)

    for supp in idx_support:
        assert supp in [
            train_dataset.data[idx]
            for idx in train_dataset.get_index_in_class(class_to_search_for)
        ], f"support {supp} does not belong to the class {class_to_search_for}"

    for idx in index_in_class:
        if train_dataset.data[idx] not in idx_support:
            test_dataset.add_datapoint(
                train_dataset.data[idx], train_dataset.labels_str[class_to_search_for]
            )

    if remove:

        idx_to_remove = []
        for idx in index_in_class:
            if train_dataset.data[idx] not in idx_support:
                idx_to_remove.append(train_dataset.data.index(train_dataset.data[idx]))

        assert len(idx_to_remove) < len(
            index_in_class
        ), f"class {class_to_search_for} will be deleted"

        assert len(idx_to_remove) + len(idx_support) == len(
            index_in_class
        ), f"{class_to_search_for} {len(idx_to_remove)} , {len(idx_support)}  == {len(index_in_class)}"


        train_dataset.remove_datapoints(idx_to_remove)

    test_dataset.update_classes_indexes()
    train_dataset.update_classes_indexes()

    if remove :
        assert train_dataset.get_index_in_class(class_to_search_for).shape[0] == len(
            idx_support
        ), f"something wrong with class length {class_to_search_for}"


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


def plot_image_to_find(class_to_search_for, test_dataset, relation, top,max_len=10):

    index_to_find = test_dataset.get_index_in_class(class_to_search_for)
    max_len = min(max_len,len(index_to_find)-1)

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
    plot=False,
    tqdm_silent=False,
    model_type="RelationNet",
    max_len = 10
):
    support_img = torch.stack([train_dataset[idx]["img"] for idx in idx_support])

    if model_type == "RelationNet":
        model_adaptater = RelationNetAdaptater(
            model, 1, len(support_img), 1, batch_size, device
        )

    elif model_type == "StandardNet":
        model_adaptater = StandardNetAdaptater(model, device)

    top, relation = model_adaptater.search_tensor(
        test_taskloader, support_img, class_to_search_for, tqdm_silent=tqdm_silent
    )

    if plot:
        imshow(support_img[0])

    c20 = count_in_top(top[:20], class_to_search_for, test_dataset)
    c100 = count_in_top(top[:100], class_to_search_for, test_dataset)
    c5 = count_in_top(top[:5], class_to_search_for, test_dataset)
    c1000 = count_in_top(top[:1000], class_to_search_for, test_dataset)

    if plot:
        plot_search(
            50,
            top,
            relation,
            test_dataset,
            figsize=(15, 15),
            ncols=6,
        )

        plot_image_to_find(class_to_search_for, test_dataset, relation, top,max_len=max_len)

    return (
        len(test_dataset.get_index_in_class(class_to_search_for)),
        c5,
        c20,
        c100,
        c1000,
        top,
        relation,
    )


### active_loops:


import sklearn.metrics as metrics

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


def found_new_images(
    model_adapter,
    class_,
    test_taskloader,
    train_dataset,
    top_to_select=3,
):
    top, _ = model_adapter.search_tensor(
        test_taskloader,
        train_dataset.get_index_in_class(class_),
        class_,
        tqdm_silent=False,
    )
    topX = top[:top_to_select][:, 0]

    datapoint_to_add = {}

    for data in topX:

        class_data = test_taskloader.dataset[data]["label"]

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
    top_to_select=1,
    treshold=0.5,
    only_true_image=True,
    checkpoint=False,
    nb_of_eval=1,
    search=False,
):

    trainer.fit(
        epochs,
        nb_of_eval,
        optim_resnet,
        scheduler_resnet,
        train_loader,
        val_loader,
        silent=False,
    )

    if checkpoint:
        trainer.model_adaptater.model = trainer.model_checkpoint


    outputs, true_labels = trainer.get_all_outputs(val_loader, silent=True)

    precision = torch.Tensor(
        metrics.precision_score(true_labels.to("cpu"), outputs.to("cpu"), average=None,zero_division=0)

    )
    recall = torch.Tensor(
        metrics.recall_score(
            true_labels.to("cpu"), outputs.to("cpu"), average=None, zero_division=0
        )
    )

    precision_mask = precision[mask]
    recall_mask = recall[mask]

    if search:
        class_to_rebalanced = mask[torch.where(precision[mask] <= treshold)[0]]

        for class_ in class_to_rebalanced:
            datapoint_to_add = found_new_images(
                trainer.model_adaptater,
                class_,
                test_taskloader,
                train_loader.dataset,
                top_to_select=top_to_select,
            )
            move_found_images(datapoint_to_add, train_loader.dataset, test_taskloader.dataset)

    return precision_mask, recall_mask
