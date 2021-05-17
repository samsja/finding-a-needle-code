import torch
import torch.nn as nn

import numpy as np
from sklearn.datasets import make_blobs, make_moons
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_and_vizu(X, y, lr, epochs):
    data = torch.from_numpy(X).float().to(device)
    labels = torch.from_numpy(y).long().to(device)

    model, acc = train(data, labels, lr, epochs)
    print(acc)

    vizu(model, X, y)

    return model

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


def train(data,labels,lr=1e-2, epochs=100):
    model = nn.Sequential(
        nn.Linear(2, len(labels.unique())),
    )

    model = model.to(device)
    
    return continue_train(model,data, labels, lr=1e-2, epochs=100)

def continue_train(model,data, labels, lr=1e-2, epochs=100):
    loss_fn = nn.CrossEntropyLoss()
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    
    for _ in range(epochs):

        model.zero_grad()
        outputs = model(data)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optim.step()

    with torch.no_grad():

        _, outputs = model(data).max(dim=1)

        acc = (outputs == labels).float().mean()

    return model, acc

def vizu(model, X, y, figsize=(17, 7)):
    h = 0.25
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
