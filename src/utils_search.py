import torch
from tqdm import tqdm

from src.few_shot_learning.relation_net import (
    RelationNet,
    RelationNetAdaptater,
    get_relation_net_adaptater,
    get_features_for_one_class,
)

from src.utils_plot import img_from_tensor, imshow, plot_list


def prepare_dataset(class_to_search_for, idx_support, train_dataset, test_dataset):
    for idx in train_dataset.get_index_in_class(class_to_search_for):
        if idx not in idx_support:
            test_dataset.add_datapoint(
                train_dataset.data[idx], train_dataset.labels_str[class_to_search_for]
            )

    test_dataset.update_classes_indexes()


def count_in_top(top, class_, test_dataset):
    top_labels = [test_dataset[idx]["label"].item() for idx in top]
    return top_labels.count(class_)


def plot_search(
    n, top, relation, test_dataset, figsize=(7, 7), img_from_tensor=img_from_tensor, ncols=4
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



def plot_image_to_find(class_to_search_for,test_dataset,relation,top):

    index_to_find = test_dataset.get_index_in_class(class_to_search_for)

    top = top.squeeze(1)
    relation = torch.stack([relation[top==i] for i in index_to_find])

    title = ["{:.2f}".format(r.item()) for r in relation ]

    target_images = torch.stack([test_dataset[i]["img"] for i in index_to_find])

    plot_list(target_images,title=title)

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
):
    support_img = torch.stack([train_dataset[idx]["img"] for idx in idx_support])

    model_adaptater = RelationNetAdaptater(
        model, 1, len(support_img), 1, batch_size, device
    )

    top, relation = model_adaptater.search(
        test_taskloader, support_img, None, tqdm_silent=tqdm_silent
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
        
        plot_image_to_find(class_to_search_for,test_dataset,relation,top)

    
    return len(test_dataset.get_index_in_class(class_to_search_for)), c5,c20, c100,c1000, top,relation
