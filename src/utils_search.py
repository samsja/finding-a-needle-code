import torch
from tqdm import tqdm

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

        train_dataset.remove_datapoints(idx_to_remove)

    test_dataset.update_classes_indexes()
    train_dataset.update_classes_indexes()


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


def plot_image_to_find(class_to_search_for, test_dataset, relation, top):

    index_to_find = test_dataset.get_index_in_class(class_to_search_for)

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

        plot_image_to_find(class_to_search_for, test_dataset, relation, top)

    return (
        len(test_dataset.get_index_in_class(class_to_search_for)),
        c5,
        c20,
        c100,
        c1000,
        top,
        relation,
    )
