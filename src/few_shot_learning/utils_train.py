import torchvision
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
from tqdm import tqdm
import random
import copy
from typing import Tuple

class RotationTransform:
    """Rotate by one of the given angles."""

    def __init__(self, angles, fill):
        self.angles = angles
        self.fill = fill

    def __call__(self, x):
        angle = random.choice(self.angles)
        return TF.rotate(x, angle, fill=self.fill)


class ModuleAdaptater:
    """
    Base adapater class for torch custom module to TrainerFewShot
    """

    def __init__(self, model: nn.Module):
        self.model = model

    def get_loss_and_accuracy(self, inputs, labels, accuracy=False):
        raise NotImplementedError


def get_n_trainable_param(model: nn.Module) -> Tuple[int,int]:
    """
    return the number of trainable parameters of a model
    
    return :
        Tuple ( mumber of trainable params, number of parameters)
    """
    L = torch.Tensor([p.requires_grad for p in model.parameters()])
    return len(L[L==True]),len(list(model.parameters()))


class TrainerFewShot:
    def __init__(
        self,
        model_adaptater: ModuleAdaptater,
        device: torch.device,
        checkpoint: bool = False
    ):
        super().__init__()

        self.model_adaptater = model_adaptater
        self.device = device

        self.checkpoint = checkpoint
        
        if not(self.checkpoint):
            self.model_checkpoint : nn.Module = self.model_adaptater.model
        else:
            self.model_checkpoint : nn.Module = self.model_adaptater.model # temporaly fix deep copy after the init of the lazy linear

        self.best_accuracy : Tuple[int,float] = (0,0) 

        self.list_loss = []
        self.list_loss_eval = []
        self.accuracy_eval = []

    def train_model(self, inputs, labels, optim: torch.optim):

        self.model_adaptater.model.zero_grad()
        loss, _ = self.model_adaptater.get_loss_and_accuracy(inputs, labels)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(self.model_adaptater.model.parameters(), 0.5)

        optim.step()

        return loss

    def fit(
        self,
        epochs: int,
        nb_eval: int,
        optim: torch.optim,
        scheduler: torch.optim,
        bg_taskloader: torch.utils.data.DataLoader,
        eval_taskloader: torch.utils.data.DataLoader,
    ):
        period_eval = max(epochs // nb_eval, 1)

        for epoch in tqdm(range(epochs)):

            list_loss_batch = []

            self.model_adaptater.model.train()

            for batch_idx, batch in enumerate(bg_taskloader):

                inputs, labels = batch

                loss = self.train_model(inputs, labels, optim)

                list_loss_batch.append(loss.item())

            self.list_loss.append(sum(list_loss_batch) / len(list_loss_batch))

            scheduler.step()

            if epoch % period_eval == 0:

                with torch.no_grad():
                    self.model_adaptater.model.eval()

                    list_loss_batch_eval = []
                    accuracy = 0

                    for batch_idx, batch in enumerate(eval_taskloader):

                        inputs_eval, labels_eval = batch

                        (
                            loss,
                            accuracy_batch,
                        ) = self.model_adaptater.get_loss_and_accuracy(
                            inputs_eval,
                            labels_eval,
                            accuracy=True,
                        )

                        list_loss_batch_eval.append(loss.item())
                        accuracy += accuracy_batch

                    accuracy = accuracy / len(eval_taskloader)
                    self.accuracy_eval.append(accuracy.item())
                        
                    if self.checkpoint and accuracy.item() > self.best_accuracy[1]:
                        self.model_checkpoint = copy.deepcopy(self.model_adaptater.model)
                        self.best_accuracy = (epoch,accuracy.item())

                    self.list_loss_eval.append(
                        sum(list_loss_batch_eval) / len(list_loss_batch_eval)
                    )

                lr = "{:.2e}".format(scheduler.get_last_lr()[0])
                print(
                    f"epoch : {epoch} , loss_train : {self.list_loss[-1]} , loss_val : {self.list_loss_eval[-1]} , accuracy_eval = {self.accuracy_eval[-1]} ,lr : {lr} "
                )

    def accuracy(self, accuracy_taskloader: torch.utils.data.DataLoader):


        accuracy = 0

        for batch_idx, batch in enumerate(accuracy_taskloader):

            with torch.no_grad():
                self.model_adaptater.model.eval()

                inputs, labels = batch

                _, accuracy_batch = self.model_adaptater.get_loss_and_accuracy(
                    inputs, labels, accuracy=True
                )

                accuracy += accuracy_batch

        return accuracy / len(accuracy_taskloader)
   

    def restore_checkpoint(self):

        self.model_adaptater.model = self.model_checkpoint 
