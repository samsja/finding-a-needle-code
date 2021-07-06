import torch
import torchvision

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

    support_filenames = {}
    return class_to_search_on, lambda: init_dataset(
        path_data, class_to_search_on, support_filenames, N=N, limit_search=limit_search
    )

def prepare_dataset(
    class_to_search_for, idx_support, train_dataset, test_dataset, remove=False,limit_search=None
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

    if limit_search is not None:
        
        idx_to_remove = test_dataset.get_index_in_class(class_to_search_for)[:-limit_search].tolist()
        test_dataset.remove_datapoints(idx_to_remove)
        test_dataset.update_classes_indexes()
 
