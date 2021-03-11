import torchvision
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
from tqdm import tqdm
import random


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
        self._model = model

    def get_loss_and_accuracy(self, inputs, labels, accuracy=False):
        raise NotImplementedError

    @property
    def model(self):
        return self._model


class TrainerFewShot:
    def __init__(
        self,
        model_adaptater: ModuleAdaptater,
        device: torch.device,
    ):
        super().__init__()

        self.model_adaptater = model_adaptater
        self.device = device

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
