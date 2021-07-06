import torch

from torchvision.models import resnet18

from src.few_shot_learning.utils_train import TrainerFewShot
from src.few_shot_learning.datasets import FewShotDataSet



from src.few_shot_learning import RelationNet, RelationNetAdaptater
from src.few_shot_learning.relation_net import (
    BasicRelationModule,
    ResNetEmbeddingModule,
)

from src.few_shot_learning.proto_net import ProtoNet, ProtoNetAdaptater

from src.datasource import init_few_shot_dataset


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
    epochs = 250
    nb_eval = 1

    def __init__(self, device, few_shot_param,class_to_search_on):

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

    def train_searcher(self, train_dataset: FewShotDataSet, num_workers):

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
            silent=True,
        )


class ProtoNetSearcher(Searcher):

    lr = 3e-4
    epochs = 40
    nb_eval = 1

    def __init__(self, device, few_shot_param,class_to_search_on):

        self.class_to_search_on = class_to_search_on

        self.model = ProtoNet(3).cuda()

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

        self.optim = torch.optim.Adam(
            self.model.parameters(), lr=RelationNetSearcher.lr
        )
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optim, step_size=1000, gamma=0.5
        )

        self.trainer = TrainerFewShot(
            self.train_model_adapter, device, checkpoint=False
        )

    def train_searcher(self, train_dataset: FewShotDataSet, num_workers):

        few_shot_task_loader = init_few_shot_dataset(
            train_dataset, self.class_to_search_on, num_workers
        )

        self.trainer.fit(
            ProtoNetSearcher.epochs,
            ProtoNetSearcher.nb_eval,
            self.optim,
            self.scheduler,
            few_shot_task_loader,
            few_shot_task_loader,
            silent=True,
        )


