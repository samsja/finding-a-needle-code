import torch

from torchvision.models import resnet18

from thesis_data_search.few_shot_learning.utils_train import TrainerFewShot
from thesis_data_search.few_shot_learning.datasets import FewShotDataSet


from thesis_data_search.few_shot_learning import RelationNet, RelationNetAdaptater
from thesis_data_search.few_shot_learning.relation_net import (
    BasicRelationModule,
    ResNetEmbeddingModule,
)

from thesis_data_search.few_shot_learning.proto_net import ProtoNet, ProtoNetAdaptater

from thesis_data_search.datasource import init_few_shot_dataset, copy_dataset_exclude_class


from thesis_data_search.few_shot_learning.standard_net import StandardNet, StandardNetAdaptater


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
    epochs = 400
    nb_eval = 120

    def __init__(self, device, few_shot_param, class_to_search_on, debug=False):

        self.debug = debug
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
        self, train_dataset: FewShotDataSet, val_dataset: FewShotDataSet, num_workers
    ):

        few_shot_task_loader = init_few_shot_dataset(
            train_dataset, self.class_to_search_on, num_workers
        )

        nb_eval = 1 if not self.debug else RelationNetSearcher.nb_eval
        self.trainer.fit(
            RelationNetSearcher.epochs,
            nb_eval,
            self.optim,
            self.scheduler,
            few_shot_task_loader,
            few_shot_task_loader,
            silent=not (self.debug),
        )


class ProtoNetSearcher(Searcher):

    lr = 3e-4
    epochs = 400
    nb_eval = 100

    def __init__(self, device, few_shot_param, class_to_search_on, debug=False):

        self.debug = debug
        self.class_to_search_on = class_to_search_on

        self.model = ProtoNet(3, pretrain=True).to(device)

        self.model_adapter = ProtoNetAdaptater(self.model, 1, 1, 1, 1, device)

        self.train_model_adapter = ProtoNetAdaptater(
            self.model,
            few_shot_param.episodes,
            few_shot_param.sample_per_class,
            few_shot_param.classes_per_ep,
            few_shot_param.queries,
            device,
        )

        self.device = device

        self.optim = torch.optim.Adam(self.model.parameters(), lr=ProtoNetSearcher.lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optim, step_size=1000, gamma=0.5
        )

        self.trainer = TrainerFewShot(
            self.train_model_adapter, device, checkpoint=False
        )

    def train_searcher(
        self, train_dataset: FewShotDataSet, val_dataset: FewShotDataSet, num_workers
    ):

        few_shot_task_loader = init_few_shot_dataset(
            train_dataset, self.class_to_search_on, num_workers
        )

        nb_eval = 1 if not self.debug else ProtoNetSearcher.nb_eval

        self.trainer.fit(
            ProtoNetSearcher.epochs,
            nb_eval,
            self.optim,
            self.scheduler,
            few_shot_task_loader,
            few_shot_task_loader,
            silent=not (self.debug),
        )


class StandardNetSearcher(Searcher):

    lr = 1e-3
    epochs = 20
    nb_eval = epochs
    batch_size = 256

    def __init__(self, device, number_of_class, debug=False):
        self.device = device
        self.debug = debug
        self.model = StandardNet(number_of_class, pretrained=True).to(device)
        self.model_adapter = StandardNetAdaptater(self.model, device)
        self.trainer = TrainerFewShot(self.model_adapter, device, checkpoint=True)

        self.optim = torch.optim.Adam(
            self.model.parameters(), lr=self.__class__.lr
        )

        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optim, step_size=100000, gamma=0.9
        )

    def _train(
        self,
        epochs,
        nb_eval,
        batch_size,
        train_dataset: FewShotDataSet,
        eval_dataset: FewShotDataSet,
        num_workers,
    ):

        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            shuffle=True,
            num_workers=num_workers,
            batch_size=batch_size,
        )

        val_loader = torch.utils.data.DataLoader(
            eval_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
        )

        self.trainer.fit(
            epochs,
            nb_eval,
            self.optim,
            self.scheduler,
            train_loader,
            val_loader,
            silent=not (self.debug),
        )

        self.trainer.model_adaptater.model = self.trainer.model_checkpoint

    def train_searcher(
        self,
        train_dataset: FewShotDataSet,
        eval_dataset: FewShotDataSet,
        num_workers,
    ):
        self._train(
            self.__class__.epochs,
            self.__class__.nb_eval,
            self.__class__.batch_size,
            train_dataset,
            eval_dataset,
            num_workers,
        )


class CifarStandardNetSearcher(StandardNetSearcher):

    lr = 1e-4
    epochs = 60
    nb_eval = epochs
    batch_size = 256


class FreezedStandardNetSearcher(StandardNetSearcher):

    epochs = 20

    def __init__(
        self, device, number_of_class, class_to_search_on, debug=False, num_worker=8
    ):
        super().__init__(device, number_of_class, debug)
        self.class_to_search_on = class_to_search_on
        self.num_worker = 8

    def train_searcher(
        self, train_dataset: FewShotDataSet, eval_dataset: FewShotDataSet, num_workers
    ):

        train_dataset_exclude = copy_dataset_exclude_class(
            train_dataset, self.class_to_search_on
        )
        eval_dataset_exclude = copy_dataset_exclude_class(
            eval_dataset, self.class_to_search_on
        )

        super().train_searcher(train_dataset_exclude, eval_dataset_exclude, num_workers)

        self.model.resnet.fc = torch.nn.Linear(
            self.model.resnet.fc.in_features,
            self.model.resnet.fc.out_features + len(self.class_to_search_on),
        )

        self.model.freeze_mlp()
        self.trainer = TrainerFewShot(self.model_adapter, self.device, checkpoint=True)

        super()._train(
            FreezedStandardNetSearcher.epochs,
            FreezedStandardNetSearcher.epochs,
            StandardNetSearcher.batch_size,
            train_dataset,
            eval_dataset,
            num_workers,
        )
