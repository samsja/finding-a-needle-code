import copy
import pickle
from collections import namedtuple

import torch
import torchvision
from torchvision.models import resnet18

from thesis_data_search.few_shot_learning import FewShotSampler2
from thesis_data_search.few_shot_learning.datasets import (
    FewShotDataSet,
    TrafficSignDataset,
)
from thesis_data_search.few_shot_learning.utils_train import TrainerFewShot


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


def get_transform_cifar():
    transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize(32),
            torchvision.transforms.RandomCrop(30),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),
        ]
    )

    transform_test = torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize((32, 32)),
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


def copy_dataset_exclude_class(train_dataset: TrafficSignDataset, class_to_exclude):
    new_dataset = TrafficSignDataset(
        train_dataset.data,
        train_dataset.labels_str,
        train_dataset.transform,
        root_dir="",
        exclude_class=class_to_exclude,
    )

    return new_dataset


def init_few_shot_dataset(train_dataset, class_to_search_on, num_workers=5):

    train_few_shot_dataset = copy_dataset_exclude_class(
        train_dataset, class_to_exclude=class_to_search_on
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


def init_dataset_raw(path_data, synthetic: list = []):

    transform, transform_test = get_transform()

    with open("thesis_data_search/pickles/traineval_incl_partial.pkl", "rb") as f:
        train_eval = pickle.load(f)
        train_eval = [x for x in train_eval if "partial" not in x]

    with open("thesis_data_search/pickles/test_incl_partial.pkl", "rb") as f:
        test = pickle.load(f)
        test = [x for x in test if "partial" not in x]

    with open("thesis_data_search/pickles/class_list.pkl", "rb") as f:
        label_list = pickle.load(f)
        train_, val_ = [], []

    for c in label_list:
        fns = [x for x in test if c + "/" in x]
        ratio = int(len(fns) * 0.9) - 1
        train_ += fns[:ratio]
        val_ += fns[ratio:]

    train_ = train_ + synthetic

    train_dataset = TrafficSignDataset(
        train_, label_list, transform=transform, root_dir=path_data
    )

    eval_dataset = TrafficSignDataset(
        val_, label_list, transform=transform_test, root_dir=path_data
    )

    test_dataset = TrafficSignDataset(
        train_eval, transform=transform_test, root_dir=path_data, label_list=label_list
    )

    return train_dataset, eval_dataset, test_dataset


def init_dataset(
    path_data,
    class_to_search_on,
    support_filenames,
    N=1,
    limit_search=None,
    synthetic: list = [],
):

    train_dataset, eval_dataset, test_dataset = init_dataset_raw(path_data, synthetic)

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


def get_data_6_rare(path_data, N, limit_search=None):
    class_to_search_on = torch.Tensor([25, 26, 32, 95, 152, 175]).long()

    support_filenames = {
        25: [
            f"{path_data}/patches/warning--slippery-road-surface--g1/1X0IkHf4hyDyWv9jdL25uXQ.jpg"
        ],
        26: [
            f"{path_data}/patches/warning--curve-left--g1/0vsCk34R8N_nq31Hjadwt4Q.jpg"
        ],
        32: [
            f"{path_data}/patches/regulatory--no-overtaking--g2/1eP65vCRiyu_x8nOJl_otsg.jpg"
        ],
        95: [
            f"{path_data}/patches/regulatory--no-stopping--g2/1x3iZWxvj6fTaLeBiiTPeEA.jpg"
        ],
        152: [
            f"{path_data}/patches/regulatory--maximum-speed-limit-20--g1/1t7pO54Ujrat7T33j3uGTOg.jpg"
        ],
        175: [
            f"{path_data}/patches/warning--slippery-road-surface--g2/1PYQrF98Be90rnFsFBpO6Qg.jpg"
        ],
    }

    return class_to_search_on, lambda: init_dataset(
        path_data, class_to_search_on, support_filenames, N=N, limit_search=limit_search
    )


def support_to_list(support_filenames):
    support_list = []

    for class_ in support_filenames.keys():
        for filename in support_filenames[class_]:
            support_list.append(filename)
    return support_list


def add_path_data_to_support_filenames(support_filenames, path_data):

    support_filenames = copy.deepcopy(support_filenames)

    for class_ in support_filenames.keys():
        fn_with_path = []
        for fn in support_filenames[class_]:
            fn_with_path.append(f"{path_data}/{fn}")
        support_filenames[class_] = fn_with_path

    return support_filenames


def get_data_6_rare_sy(path_data, N, limit_search=None):
    class_to_search_on = torch.Tensor([25, 26, 32, 95, 152, 175]).long()

    support_filenames = {
        25: [f"artificial/warning--slippery-road-surface--g1/artificial_0.jpg"],
        26: [f"artificial/warning--curve-left--g1/artificial_0.jpg"],
        32: [f"artificial/regulatory--no-overtaking--g2/artificial_0.jpg"],
        95: [f"artificial/regulatory--no-stopping--g2/artificial_0.jpg"],
        152: [f"artificial/regulatory--maximum-speed-limit-20--g1/artificial_0.jpg"],
        175: [f"artificial/warning--slippery-road-surface--g2/artificial_0.jpg"],
    }

    return class_to_search_on, lambda: init_dataset(
        path_data,
        class_to_search_on,
        add_path_data_to_support_filenames(support_filenames, path_data),
        N=N,
        limit_search=limit_search,
        synthetic=support_to_list(support_filenames),
    )


def get_data_25_rare(path_data, N, limit_search=None):
    class_to_search_on = torch.Tensor(
        [
            276,
            311,
            295,
            312,
            255,
            263,
            309,
            254,
            299,
            290,
            307,
            146,
            275,
            176,
            178,
            143,
            234,
            33,
            187,
            259,
            20,
            284,
            223,
            271,
            273,
        ]
    ).long()

    support_filenames = {
        263: [
            f"{path_data}/patches/warning--winding-road-first-right--g3/1rPRee8UZKlRqeRGPkVmHSA.jpg"
        ],
        309: [
            f"{path_data}/patches/regulatory--bicycles-only--g2/2sMC4I8VfvZ8or_lhJv9adw.jpg"
        ],
        254: [
            f"{path_data}/patches/regulatory--no-parking-or-no-stopping--g5/43CkesMYM4L226r2cfDxeWQ.jpg"
        ],
        290: [
            f"{path_data}/patches/regulatory--bicycles-only--g3/4J8sy1kBZ3JIh7zpMglbvFQ.jpg"
        ],
        146: [f"{path_data}/patches/warning--t-roads--g2/8wG_KUY743OkU5T39SdXb1w.jpg"],
        275: [
            f"{path_data}/patches/regulatory--maximum-speed-limit-led-80--g1/2kbObURhSNEctmYjwvKwpow.jpg"
        ],
        176: [
            f"{path_data}/patches/information--interstate-route--g1/66EQvCrRMh6QLJh1rDe3TzA.jpg"
        ],
        178: [
            f"{path_data}/patches/information--airport--g1/9J5YUPUr_3hNc4MXgy-IfIw.jpg"
        ],
        234: [
            f"{path_data}/patches/regulatory--triple-lanes-turn-left-center-lane--g1/2XkDjCZm8sxwszkTQEqHWtw.jpg"
        ],
        20: [
            f"{path_data}/patches/regulatory--pedestrians-only--g2/5xILijMFXCO1Jm9qSyb78TA.jpg"
        ],
        271: [
            f"{path_data}/patches/complementary--buses--g1/7nT5RAgHRq8k5dow-OQA-xw.jpg"
        ],
        273: [
            f"{path_data}/patches/information--telephone--g1/2gQ4_1gp-V9YSuXL7WDf9eQ.jpg"
        ],
    }
    return class_to_search_on, lambda: init_dataset(
        path_data, class_to_search_on, support_filenames, N=N, limit_search=limit_search
    )


def get_data_cifar(path_data, N, limit_search=None):
    with open("thesis_data_search/pickles/cifar_100/train.pkl", "rb") as f:
        list_train = pickle.load(f)

    with open("thesis_data_search/pickles/cifar_100/eval.pkl", "rb") as f:
        list_eval = pickle.load(f)

    with open("thesis_data_search/pickles/cifar_100/test.pkl", "rb") as f:
        list_test = pickle.load(f)

    with open("thesis_data_search/pickles/cifar_100/label_list.pkl", "rb") as f:
        label_list = pickle.load(f)

    transform, transform_test = get_transform_cifar()
    train_dataset = TrafficSignDataset(
        list_train, label_list, transform=transform, root_dir=path_data
    )
    eval_dataset = TrafficSignDataset(
        list_eval, label_list, transform=transform_test, root_dir=path_data
    )
    test_dataset = TrafficSignDataset(
        list_test, label_list, transform=transform_test, root_dir=path_data
    )

    class_to_search_on = [81, 14, 3, 94, 35, 31, 28, 17, 13, 86]

    for class_ in class_to_search_on:
        train_dataset.remove_datapoints(train_dataset.get_index_in_class(class_)[N:])
        train_dataset.update_classes_indexes()

        assert len(train_dataset.get_index_in_class(class_)) == N

    n_test = 20
    for class_ in class_to_search_on:
        test_dataset.remove_datapoints(test_dataset.get_index_in_class(class_)[n_test:])
        test_dataset.update_classes_indexes()

        assert len(test_dataset.get_index_in_class(class_)) == n_test

    def get_dataset():
        return train_dataset, eval_dataset, test_dataset

    return class_to_search_on, get_dataset


def prepare_dataset(
    class_to_search_for,
    idx_support,
    train_dataset,
    test_dataset,
    remove,
    limit_search=None,
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

    if remove:
        assert train_dataset.get_index_in_class(class_to_search_for).shape[0] == len(
            idx_support
        ), f"something wrong with class length {class_to_search_for}"

    if limit_search is not None:

        if limit_search < len(test_dataset.get_index_in_class(class_to_search_for)):
            idx_to_remove = test_dataset.get_index_in_class(class_to_search_for)[
                :-limit_search
            ].tolist()
            test_dataset.remove_datapoints(idx_to_remove)
            test_dataset.update_classes_indexes()
