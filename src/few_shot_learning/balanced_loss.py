import torch
import torch.nn as nn


def balanced_loss(outputs, labels):

    distrib = labels.bincount()

    loss_fn = nn.CrossEntropyLoss()

    loss = torch.tensor(0).float().to(outputs.device)
    loss.requires_grad = True

    for class_, count in enumerate(distrib):

        loss_class = (
            loss_fn(outputs[labels == class_], labels[labels == class_])
            * (sum(distrib) / count)
            if count != 0
            else torch.tensor(0).float().to(outputs.device)
        )

        loss = loss + loss_class

        return loss
