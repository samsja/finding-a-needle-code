from src.few_shot_learning.utils_train import TrainerFewShot
from src.datasource import FewShotParam
from src.few_shot_learning.relation_net import (
    RelationNet,
    RelationNetAdaptater,
    VeryBasicRelationModule,
)
import torch
import torch.nn as nn

import numpy as np
from sklearn.datasets import make_blobs, make_moons
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

import ipywidgets as widgets
from ipywidgets import fixed, interact, interact_manual, interactive
import seaborn as sns

def make_blob_torch(n_samples, centers, cluster_std, random_state, ratio, device):
    X, y = make_blobs_few_shot(
        *make_blobs(
            n_samples=n_samples,
            centers=centers,
            cluster_std=cluster_std,
            random_state=random_state,
        ),
        ratio,
    )

    data = torch.from_numpy(X).float().to(device)
    labels = torch.from_numpy(y).long().to(device)

    return data, labels


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


from src.few_shot_learning.datasets import FewShotDataSet
from src.few_shot_learning.sampler import FewShotSampler2


class BlobFSDataSet(FewShotDataSet):
    def __init__(self, data, labels):
        super().__init__()

        self.data = data
        self.labels = labels
        self._classes = labels.unique().to("cpu")

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

    def get_index_in_class(self, class_idx: int):
        """
        Method to get the indexes of the elements in the same class as class_idx

        # Args:
            class_idx : int. The index of the desider class

        """
        return torch.where(self.labels == class_idx)[0]

    def get_index_in_class_vect(self, class_idx):

        return [self.get_index_in_class(c.item()) for c in class_idx]


def train_few_shot(
    model,
    data,
    labels,
    device,
    lr=1e-2,
    epochs=1000,
    balanced_loss=False,
    callback=None,
):
    few_shot_dataset = BlobFSDataSet(data.to(device), labels.to(device))
    few_shot_param = FewShotPa120m(1, 1, 1, len(labels.unique()), 10)

    sampler = FewShotSampler2(few_shot_dataset, *few_shot_param)

    few_shot_taskloader = torch.utils.data.DataLoader(
        few_shot_dataset, batch_sampler=sampler, num_workers=0
    )

    model_adapter = RelationNetAdaptater(
        model,
        few_shot_param.episodes,
        few_shot_param.sample_per_class,
        few_shot_param.classes_per_ep,
        few_shot_param.queries,
        device,
    )

    train_model_adapter = RelationNetAdaptater(
        model,
        few_shot_param.episodes,
        few_shot_param.sample_per_class,
        few_shot_param.classes_per_ep,
        few_shot_param.queries,
        device,
    )

    device = device

    optim = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=1000, gamma=0.5)

    trainer = TrainerFewShot(train_model_adapter, device, checkpoint=False)

    trainer.fit(
        epochs,
        1,
        optim,
        scheduler,
        few_shot_taskloader,
        few_shot_taskloader,
        silent=False,
    )

    return model


def train_custom_loss(model, data, labels, lr=1e-2, epochs=100, callback=None):
    loss_fn = nn.CrossEntropyLoss()
    optim = torch.optim.SGD(model.parameters(), lr=lr)

    for _ in tqdm(range(epochs)):

        model.zero_grad()
        outputs = model(data)
        loss = 0

        distrib = labels.bincount()

        for class_, count in enumerate(distrib):

            loss += loss_fn(outputs[labels == class_], labels[labels == class_]) * (
                sum(distrib) / count
            )

        loss.backward()
        optim.step()

    return model


def train_normal_loss(model, data, labels, lr=1e-2, epochs=100, callback=None):
    loss_fn = nn.CrossEntropyLoss()
    optim = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)

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

    with torch.no_grad():
        _, Z = model(torch.from_numpy(Xmesh).float().to(device)).max(dim=1)

    Z = Z.to("cpu").numpy().reshape(xx.shape)

    fig = plt.figure(figsize=figsize)
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral, alpha=0.8)
    plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.Spectral)

    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())


def vizu_proba(model, X, y, device, selection_data, selection_labels, figsize=(5, 5),space=(-1,1,-1,1)):
    h = 0.05
    X = X.to("cpu")
    y = y.to("cpu")
    x_min, x_max = X[:, 0].min() + space[0], X[:, 0].max() + space[1]
    y_min, y_max = X[:, 1].min() + space[2], X[:, 1].max() + space[3]
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Xmesh = np.c_[xx.ravel(), yy.ravel()]

    with torch.no_grad():
        Z = model(torch.from_numpy(Xmesh).float().to(device))
    if Z.shape[1] > 1:
        Z = Z.softmax(dim=1)[:, 0]

    Z = Z.to("cpu").numpy().reshape(xx.shape)

    fig = plt.figure(figsize=figsize)
    cs = plt.contourf(xx, yy, Z, cmap=plt.cm.get_cmap("Greys"), alpha=0.8)
    plt.colorbar(cs)

    mask = np.where(y!=0)
    mask_rare = np.where(y==0)
    colormap = np.array(sns.color_palette("hls",8).as_hex())

    plt.scatter(X[mask, 0], X[mask, 1], c=colormap[y[mask]], s=6,marker="o")
    plt.scatter(X[mask_rare, 0], X[mask_rare, 1], c=colormap[y[mask_rare]], s=15,marker="o")
    if selection_data is not None:
        plt.scatter(
            selection_data.to("cpu")[:, 0],
            selection_data.to("cpu")[:, 1],
            s=12,
            marker="s",
            facecolors="none",
            edgecolor=colormap[selection_labels.to("cpu")],
            #  c=colormap[selection_labels.to("cpu")],
        )

    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())


def boxplot_proba_few_shot(model, X, y, device):

    with torch.no_grad():
        outputs = model(X).softmax(dim=1)

    plt.boxplot([outputs[y == 0][:, 0].to("cpu"), outputs[y != 0][:, 0].to("cpu")])
    plt.title("Probability that point belongs to the class with few shot")


class ConvUnsqueezer(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.view(1, len(x), 2, 1, 1)


def get_protonet_model(x, y, rare_class_index=0):
    rare_x = x[y == rare_class_index]
    prototype = rare_x.mean(dim=0)

    class ProtonetModel(torch.nn.Module):
        def __init__(self, prototype):
            super().__init__()
            self.prototype = prototype

        def forward(self, x):
            return -torch.cdist(x, self.prototype.unsqueeze(0))

    return ProtonetModel(prototype)


def get_relationnet_model(x, y, device, rare_class_index=0):
    relation_net = RelationNet(
        device=device,
        out_channels=1,
        in_channels=2,
        embedding_module=ConvUnsqueezer(),
        relation_module=VeryBasicRelationModule(input_size=2),
    ).to(device)

    relation_net = train_few_shot(relation_net, x, y, device)

    rare_x = x[y == rare_class_index]
    prototype = rare_x.mean(dim=0)

    class SearchingRelationNet(torch.nn.Module):
        def __init__(self, relation_net, prototype):
            super().__init__()
            self.relation_net = relation_net
            self.prototype = prototype

        def forward(self, x):
            inp = torch.cat((self.prototype.unsqueeze(0), x))
            # breakpoint()
            result = self.relation_net(inp, 1, 1, 1, len(x))
            return result

    return SearchingRelationNet(relation_net, prototype)


def get_main(device, holder):
    @interact(
        n_train_samples=widgets.IntSlider(
            min=100, max=2000, step=100, value=500, continuous_update=False
        ),
        n_selection_samples=widgets.IntSlider(
            min=0, max=2000, step=100, value=800, continuous_update=False
        ),
        centers=widgets.IntSlider(
            min=2, max=10, step=1, value=3, continuous_update=False
        ),
        cluster_std=widgets.FloatSlider(
            min=0, max=1, step=0.1, value=0.5, continuous_update=False
        ),
        ratio=widgets.FloatSlider(
            min=0, max=1, step=0.01, value=0.02, continuous_update=False
        ),
        lr=widgets.FloatText(value=1e-2, description="lr:", disabled=False),
        epochs=widgets.IntText(value=100, description="epochs:", disabled=False),
        balanced_loss=widgets.Checkbox(
            value=False, description="Balanced_loss", disabled=False, indent=False
        ),
        dist_common=widgets.FloatSlider(
            min=0, max=2, step=0.1, value=1, continuous_update=False
        ),
        dist_rare=widgets.FloatSlider(
            min=0, max=2, step=0.1, value=1, continuous_update=False
        ),
        proto_net=widgets.Checkbox(
            value=False, description="ProtoNet", disabled=False, indent=False
        ),
        relation_net=widgets.Checkbox(
            value=False, description="RelationNet", disabled=False, indent=False
        ),
        zoom=widgets.Checkbox(
            value=True, description="Zoom", disabled=False, indent=False
        ),
    )
    def main(
        n_train_samples=1000,
        n_selection_samples=0,
        centers=2,
        dist_common=1,
        dist_rare=1,
        cluster_std=0.5,
        ratio=1,
        lr=1e-2,
        epochs=100,
        balanced_loss=False,
        proto_net=False,
        relation_net=False,
        zoom = True,
    ):


        space = (-0.5,0.5,-0.5,0.1) if zoom else  (-10,10,-10,2)
        
        centers = [[0, 0], [-dist_common, dist_rare], [dist_common, dist_rare]]

        rand_state = np.random.RandomState(4) #2

        holder.data, holder.labels = make_blob_torch(
            n_train_samples, centers, cluster_std, rand_state, ratio, device
        )

        holder.model, acc = train(
            holder.data, holder.labels, device, lr, epochs, balanced_loss
        )

        selection_data, selection_labels = make_blob_torch(
            2*int(n_selection_samples*ratio), centers, cluster_std, rand_state, 0.5, device
        )

        vizu_proba(
            holder.model,
            holder.data,
            holder.labels,
            device,
            selection_data,
            selection_labels,
            space = space, 
        )

        #  plt.title("StandardNet selection function")
        plt.show()


        if proto_net:
            protonet_model = get_protonet_model(holder.data, holder.labels)
            vizu_proba(
                protonet_model,
                holder.data,
                holder.labels,
                device,
                selection_data,
                selection_labels,
            )
            plt.title("ProtoNet selection function")
            plt.show()

        if relation_net:
            relationnet_model = get_relationnet_model(
                holder.data, holder.labels, device
            )
            vizu_proba(
                relationnet_model,
                holder.data,
                holder.labels,
                device,
                selection_data,
                selection_labels,
            )
            plt.title("RelationNet selection function")
            plt.show()

        # boxplot_proba_few_shot(holder.model, holder.data, holder.labels, device)
        # plt.show()

    return main


### sympy


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
