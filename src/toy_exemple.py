import torch
import torch.nn as nn

import numpy as np
from sklearn.datasets import make_blobs, make_moons
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

def make_blob_torch(n_samples,centers,cluster_std,random_state,ratio,device):
    X, y = make_blobs_few_shot(
        *make_blobs(
            n_samples=n_samples,
            centers=centers,
            cluster_std=cluster_std,
            random_state=random_state,
        ),
        ratio
    )

    data = torch.from_numpy(X).float().to(device)
    labels = torch.from_numpy(y).long().to(device) 

    return data,labels

def make_blobs_few_shot(X, y, ratio):
    n = max(int(X[y == 0].shape[0] * ratio), 1)

    classes = np.unique(y)

    X_few = np.concatenate(
        [X[y == 0][0:n]] + [X[y == class_] for class_ in classes if class_ != 0]
    )
    y_few = np.concatenate(
        [y[y == 0][0:n]] + [y[y == class_] for class_ in classes if class_ != 0]
    )

    return X_few, y_few


def train(
    data, labels, device, lr=1e-2, epochs=100, balanced_loss=False, callback=None
):
    model = nn.Sequential(
        nn.Linear(2, len(labels.unique())),
    )

    model = model.to(device)

    if balanced_loss:
        model = train_custom_loss(model, data, labels, lr, epochs, callback)
    else:
        model = train_normal_loss(model, data, labels, lr, epochs, callback)

    with torch.no_grad():

        _, outputs = model(data).max(dim=1)

        acc = (outputs == labels).float().mean()

    return model, acc


def train_custom_loss(model, data, labels, lr=1e-2, epochs=100, callback=None):
    loss_fn = nn.CrossEntropyLoss()
    optim = torch.optim.SGD(model.parameters(), lr=lr)

    for _ in tqdm(range(epochs)):

        model.zero_grad()
        outputs = model(data)
        loss = 0

        distrib = labels.bincount()

        for class_, count in enumerate(distrib):

            loss += loss_fn(outputs[labels == class_], labels[labels == class_])* ( sum(distrib)/ count)

        loss.backward()
        optim.step()

    return model


def train_normal_loss(model, data, labels, lr=1e-2, epochs=100, callback=None):
    loss_fn = nn.CrossEntropyLoss()
    optim = torch.optim.SGD(model.parameters(), lr=lr,momentum=0.9)

    for idx_epoch in tqdm(range(epochs)):

        model.zero_grad()
        outputs = model(data)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optim.step()

        if callback is not None:
            callback(
                model=model,
                data=data,
                labels=labels,
                epoch=idx_epoch,
                max_epochs=epochs,
            )

    return model


def vizu(model, X, y, device, figsize=(17, 7)):
    h = 0.05
    X = X.to("cpu")
    y = y.to("cpu") 
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Xmesh = np.c_[xx.ravel(), yy.ravel()]

    _, Z = model(torch.from_numpy(Xmesh).float().to(device)).max(dim=1)

    Z = Z.to("cpu").numpy().reshape(xx.shape)

    fig = plt.figure(figsize=figsize)
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral, alpha=0.8)
    plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.Spectral)

    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())


def boxplot_proba_few_shot(model, X, y, device):

    with torch.no_grad():
        outputs = model(X).softmax(dim=1)

    plt.boxplot([outputs[y == 0][:, 0].to("cpu"), outputs[y != 0][:, 0].to("cpu")])
    plt.title("Probability that point belongs to the class with few shot")


import sympy as sp

class SymbolicNN:
    def __init__(self, N):

        self.N = N

        self.w = []
        self.b = []

        for i in range(self.N):
            self.w.append(sp.Symbol(f"w_{i}"))
            self.b.append(sp.Symbol(f"b_{i}"))

    def a(self, x, i, mu):
        return self.w[i] * x[mu] + self.b[i]

    def o(self, x, i, mu):
        return sp.exp(self.a(x, i, mu)) / (
            sum(sp.exp(self.a(x, j, mu)) for j in range(self.N))
        )

    def loss(self, x):
        return -sum([sp.log(self.o(x, mu, mu)) for mu in range(self.N)])

    def loss_w(self, x, weight):
        return -sum([weight[mu] * sp.log(self.o(x, mu, mu)) for mu in range(self.N)])

    def regul(self):
        return sp.sqrt(sum([w * w for w in self.w]))

    def loss_regul(self, x):
        return self.loss(x) + self.regul()

    def accuracy(self, x, y, w, b):

        if self.N != 2:
            raise NotImplementedError("N should be equal to 2 ")

        subs = {model.w[i]: w[i] for i in range(len(model.w))}
        subs.update({model.b[i]: b[i] for i in range(len(model.b))})

        def acc(mu):
            return 1 if (self.o(x, mu, mu) > self.o(x, 1 - mu, mu)).subs(subs) else 0

        return sum([acc(y) for y in Y])
