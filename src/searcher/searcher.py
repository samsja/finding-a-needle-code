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


from src.few_shot_learning.standard_net import StandardNet, StandardNetAdaptater

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
    nb_eval = 120

    def __init__(self, device, few_shot_param,class_to_search_on,debug=False):

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

    def train_searcher(self, train_dataset: FewShotDataSet, val_dataset: FewShotDataSet, num_workers):

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
            silent=not(self.debug),
        )


class ProtoNetSearcher(Searcher):

    lr = 3e-4
    epochs = 200
    nb_eval = 100

    def __init__(self, device, few_shot_param,class_to_search_on,debug=False):

        self.debug = debug
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

    def train_searcher(self, train_dataset: FewShotDataSet, val_dataset: FewShotDataSet, num_workers):

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
            silent=not(self.debug),    
        )


class StandardNetSearcher(Searcher):

    lr = 1e-3
    epochs = 1
    nb_eval = epochs
    batch_size = 256

    def __init__(self, device,number_of_class,debug=False):
        self.debug = debug
        self.model = StandardNet(number_of_class).to(device)
        self.model_adapter = StandardNetAdaptater(self.model, device)
        self.trainer = TrainerFewShot(self.model_adapter,device, checkpoint=True)

        self.optim = torch.optim.Adam(self.model.parameters(), lr=StandardNetSearcher.lr)

        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optim, step_size=100000, gamma=0.9
        )




    def train_searcher(self, train_dataset: FewShotDataSet, eval_dataset: FewShotDataSet, num_workers):

        train_loader = torch.utils.data.DataLoader(
            train_dataset, shuffle=True, num_workers=num_workers, batch_size=StandardNetSearcher.batch_size
        )

        val_loader = torch.utils.data.DataLoader(
            eval_dataset, batch_size=StandardNetSearcher.batch_size, num_workers=num_workers
        )

        self.trainer.fit(
            StandardNetSearcher.epochs,
            StandardNetSearcher.nb_eval,
            self.optim,
            self.scheduler,
            train_loader,
            val_loader,
            silent=not(self.debug),    
        )
        
        self.trainer.model_adaptater.model = self.trainer.model_checkpoint



