import torchvision
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
from tqdm import tqdm
import random
import copy
from typing import Iterator, Tuple


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

    def get_loss_and_accuracy(self, inputs, labels, accuracy=False) -> Tuple[float,float]:
        raise NotImplementedError


def get_n_trainable_param(model: nn.Module) -> Tuple[int, int]:
    """
    return the number of trainable parameters of a model

    return :
        Tuple ( mumber of trainable params, number of parameters)
    """
    L = torch.Tensor([p.requires_grad for p in model.parameters()])
    return len(L[L == True]), len(list(model.parameters()))


class TrainerFewShot:
    def __init__(
        self,
        model_adaptater: ModuleAdaptater,
        device: torch.device,
        checkpoint: bool = False,
        clip_grad: bool =  True,
       
    ):
        super().__init__()

        self.model_adaptater = model_adaptater
        self.device = device

        self.checkpoint = checkpoint

        if not (self.checkpoint):
            self.model_checkpoint: nn.Module = self.model_adaptater.model
        else:
            self.model_checkpoint: nn.Module = (
                self.model_adaptater.model
            )  # temporaly fix deep copy after the init of the lazy linear

        self.best_accuracy: Tuple[int, float] = (0, 0)

        self.list_loss = []
        self.list_loss_eval = []
        self.accuracy_eval = []

        self.clip_grad = clip_grad

       
    def train_model(self, inputs, labels, optim: torch.optim):

        self.model_adaptater.model.zero_grad()
        loss, _ = self.model_adaptater.get_loss_and_accuracy(inputs, labels)

        loss.backward()

        if self.clip_grad:
            torch.nn.utils.clip_grad_norm_(self.model_adaptater.model.parameters(), 0.5)

        optim.step()

        return loss


    def _get_data_from_batch(self,batch):
        
        try:
            inputs,labels = batch["img"],batch["label"]
        except TypeError:
            inputs, labels = batch


        return inputs,labels


    def fit(
        self,
        epochs: int,
        nb_eval: int,
        optim: torch.optim,
        scheduler: torch.optim,
        bg_taskloader: torch.utils.data.DataLoader,
        eval_taskloader: torch.utils.data.DataLoader,
        tqdm_on_batch = False,
    ):
        period_eval = max(epochs // nb_eval, 1)

        if tqdm_on_batch:
            enumerate2 = lambda x : enumerate(tqdm(x))
            range2 =  lambda x : range(x)
        else:
            enumerate2 = lambda x : enumerate(x)
            range2 = lambda x : tqdm(range(x))

        for epoch in range2(epochs):

            list_loss_batch = []

            self.model_adaptater.model.train()

            for batch_idx, batch in enumerate2(bg_taskloader):

                inputs, labels = self._get_data_from_batch(batch)

                loss = self.train_model(inputs.to(self.device), labels.to(self.device), optim)

                list_loss_batch.append(loss.item())

            self.list_loss.append(sum(list_loss_batch) / len(list_loss_batch))

            scheduler.step()

            if epoch % period_eval == 0:

                with torch.no_grad():
                    self.model_adaptater.model.eval()

                    list_loss_batch_eval = []
                    accuracy = 0

                    for batch_idx, batch in enumerate(eval_taskloader):

                        inputs_eval, labels_eval = self._get_data_from_batch(batch)

                        (
                            loss,
                            accuracy_batch,
                        ) = self.model_adaptater.get_loss_and_accuracy(
                            inputs_eval.to(self.device),
                            labels_eval.to(self.device),
                            accuracy=True,
                        )

                        list_loss_batch_eval.append(loss.item())
                        accuracy += accuracy_batch

                    accuracy = accuracy / len(eval_taskloader)
                    self.accuracy_eval.append(accuracy.item())

                    if self.checkpoint and accuracy.item() > self.best_accuracy[1]:
                        self.model_checkpoint = copy.deepcopy(
                            self.model_adaptater.model
                        )
                        self.best_accuracy = (epoch, accuracy.item())

                    self.list_loss_eval.append(
                        sum(list_loss_batch_eval) / len(list_loss_batch_eval)
                    )

                lr = "{:.2e}".format(scheduler.get_last_lr()[0])
                print(
                    f"epoch : {epoch} , loss_train : {self.list_loss[-1]} , loss_val : {self.list_loss_eval[-1]} , accuracy_eval = {self.accuracy_eval[-1]} ,lr : {lr} "
                )

    def accuracy(self, accuracy_taskloader: torch.utils.data.DataLoader):

        accuracy = 0

        for batch_idx, batch in enumerate(tqdm(accuracy_taskloader)):

            with torch.no_grad():
                self.model_adaptater.model.eval()

                inputs, labels = self._get_data_from_batch(batch)

                _, accuracy_batch = self.model_adaptater.get_loss_and_accuracy(
                    inputs.to(self.device), labels.to(self.device), accuracy=True
                )

                accuracy += accuracy_batch

        return accuracy / len(accuracy_taskloader)

    def restore_checkpoint(self):

        self.model_adaptater.model = self.model_checkpoint


def get_miss_match_few_shot(
    model_adaptater: ModuleAdaptater, accuracy_taskloader: torch.utils.data.DataLoader
) -> Iterator[
    Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]
]:
    """
    this function help to find example where the model make a wrong guess

    #return:
        iterator compose of tuple (miss match queries images, miss match queries output label, miss match queries true label,support set images,relation value output, relation value true target )

    """
    if model_adaptater.nb_ep != 1:
        raise NotImplementedError

    for batch_idx, batch in enumerate(accuracy_taskloader):

        with torch.no_grad():
            model_adaptater.model.eval()

            try :
                inputs, _ = batch
            except:
                inputs = batch["img"]

            # mis_inputs,mis_class,true_class,support_inputs = model_adaptater.get_mismatch_inputs(inputs)

        yield model_adaptater.get_mismatch_inputs(inputs)


import matplotlib.pyplot as plt


def plot_miss_match(miss_match_tuple, img_from_tensor=lambda x: x, figsize=(7, 7)):
    """
    plot support and missmatch queries
    """

    (
        miss_inputs,
        miss_class,
        true_class,
        support_inputs,
        r_output,
        r_true_label,
    ) = miss_match_tuple

    if support_inputs.size(0) != 1:
        raise NotImplementedError

    max_miss_class = (
        torch.bincount(miss_class).max().item() if miss_class.size(0) > 0 else 0
    )

    fig = plt.figure(constrained_layout=True)

    fig1, ax1 = plt.subplots(
        nrows=support_inputs.size(1),
        ncols=support_inputs.size(2) + max_miss_class,
        constrained_layout=True,
        figsize=figsize,
    )

    for i, img_class in enumerate(support_inputs[0]):
        for j, img in enumerate(img_class):
            ax1[i, j].axis("off")
            ax1[i, j].imshow(img_from_tensor(img))

    for i in range(ax1.shape[0]):
        for j in range(ax1.shape[1]):
            ax1[i, j].axis("off")

    list_available = {i: 0 for i in range(support_inputs.size(1))}

    for j, img in enumerate(miss_inputs):

        row = int(miss_class[j])

        col = support_inputs.size(2) + list_available[row]

        ax1[row, col].axis("off")
        ax1[row, col].imshow(img_from_tensor(img))

        ro = float("{:.2f}".format(r_output[j]))
        rt = float("{:.2f}".format(r_true_label[j]))

        ax1[row, col].set_title(f"{miss_class[j]} {true_class[j]} \n {ro} {rt}")

        list_available[row] += 1
