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


class TrainerFewShot:
    def __init__(
        self,
        model: nn.Module,
        loss_func: nn.modules.loss,
        nb_ep: int,
        n: int,
        k: int,
        q: int,
        device: torch.device,
    ):
        super().__init__()

        self.model = model
        self.loss_func = loss_func
        self.nb_ep = nb_ep
        self.n = n
        self.k = k
        self.q = q
        self.device = device

        self.list_loss = []
        self.list_loss_eval = []
        self.accuracy_eval = []

    def eval_model(self, inputs_eval, accuracy=False):
        """
        evaluate the model get loss and accuracy if accuracy is set to True
        """

        outputs = self.model(
            inputs_eval.to(self.device), self.nb_ep, self.n, self.k, self.q
        )
        targets = (
            torch.eye(self.k, device=self.device)
            .unsqueeze(0)
            .unsqueeze(-1)
            .expand(self.nb_ep, self.k, self.k, self.q)
        )

        loss = self.loss_func(outputs, targets)

        if accuracy:
            accuracy = (outputs.argmax(dim=1) == targets.argmax(dim=1)).float().mean()

        return loss, accuracy

    def train_model(self, inputs, optim: torch.optim):

        self.model.zero_grad()

        loss, _ = self.eval_model(inputs)
        loss.backward()
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

            self.model.train()

            for batch_idx, batch in enumerate(bg_taskloader):

                inputs, labels = batch

                loss = self.train_model(inputs, optim)

                list_loss_batch.append(loss.item())

            self.list_loss.append(sum(list_loss_batch) / len(list_loss_batch))

            scheduler.step()

            if epoch % period_eval == 0:

                with torch.no_grad():
                    self.model.eval()

                    list_loss_batch_eval = []
                    accuracy = 0

                    for batch_idx, batch in enumerate(eval_taskloader):

                        inputs_eval, labels_eval = batch

                        loss, accuracy_batch = self.eval_model(
                            inputs_eval,
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
                self.model.eval()

                inputs, labels = batch

                outputs, accuracy_batch = self.eval_model(inputs, accuracy=True)

                accuracy += accuracy_batch

        return accuracy / len(accuracy_taskloader)
