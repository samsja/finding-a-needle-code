import torchvision
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from few_shot_learning import FewShotDataSet, FewShotSampler, Omniglot, RelationNet


import argparse

##############
# Parameters #
##############
parser = argparse.ArgumentParser()
parser.add_argument("-n", default=5, type=int)
parser.add_argument("-k", default=5, type=int)
parser.add_argument("-q", default=15, type=int)
parser.add_argument("--nb_ep", default=10, type=int)
parser.add_argument("--nb_epochs", default=10, type=int)
parser.add_argument("--nb_eval", default=20, type=int)
parser.add_argument("-lr", default=1e-4, type=float)
parser.add_argument("--nb_of_batch", default=10, type=int)


args = parser.parse_args()


nb_ep = args.nb_ep
k = args.k
q = args.q
n = args.n
number_of_batch = args.nb_of_batch

transform = torchvision.transforms.Compose(
    [
        # torchvision.transforms.RandomRotation(90,expand=True),
        # torchvision.transforms.RandomRotation(180,expand=True),
        # torchvision.transforms.RandomRotation(270,expand=True),
        torchvision.transforms.Resize(28, interpolation=2),
        torchvision.transforms.ToTensor(),
    ]
)

bg_dataset = Omniglot(
    root="/staging/thesis_data_search/data",
    download=True,
    transform=transform,
    background=True,
)

eval_dataset = Omniglot(
    root="/staging/thesis_data_search/data",
    download=True,
    transform=transform,
    background=False,
)


# ## Train

# In[12]:


def apply_model(inputs_eval, model):

    targets = (
        torch.eye(k, device=device).unsqueeze(0).unsqueeze(-1).expand(nb_ep, k, k, q)
    )
    outputs = model(inputs_eval.to(device), nb_ep, n, k, q)

    return outputs, targets


def eval_model(inputs_eval, model, loss_func):

    outputs, targets = apply_model(inputs_eval, model)

    loss = loss_func(outputs, targets)

    return loss


def train_model(inputs, model, loss_func, optim):

    model.zero_grad()

    loss = eval_model(inputs, model, loss_func)

    loss.backward()
    optim.step()

    return loss


few_shot_sampler = FewShotSampler(
    bg_dataset,
    number_of_batch=number_of_batch,
    episodes=nb_ep,
    sample_per_class=n,
    classes_per_ep=k,
    queries=q,
)

few_shot_sampler_val = FewShotSampler(
    eval_dataset,
    number_of_batch=number_of_batch,
    episodes=nb_ep,
    sample_per_class=n,
    classes_per_ep=k,
    queries=q,
)


# In[15]:


bg_taskloader = torch.utils.data.DataLoader(bg_dataset, batch_sampler=few_shot_sampler)


eval_taskloader = torch.utils.data.DataLoader(
    eval_dataset, batch_sampler=few_shot_sampler_val
)


# In[26]:


loss_func = nn.MSELoss()
model = RelationNet(in_channels=1, out_channels=64, device=device, debug=True)
model.to(device)


lr = args.lr

optim = torch.optim.Adam(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=10, gamma=0.1)

list_loss = []
list_loss_eval = []

## train

epochs = args.nb_epochs
nb_eval = args.nb_eval
period_eval = max(epochs // nb_eval, 1)


for epoch in range(epochs):

    list_loss_batch = []

    model.train()

    for batch_idx, batch in enumerate(bg_taskloader):

        inputs, labels = batch

        loss = train_model(inputs, model, loss_func, optim)

        # scheduler.step()

        list_loss_batch.append(loss.item())

    list_loss.append(sum(list_loss_batch) / len(list_loss_batch))

    if epoch % period_eval == 0:

        with torch.no_grad():
            model.eval()

            list_loss_batch_eval = []

            for batch_idx, batch in enumerate(eval_taskloader):

                inputs_eval, labels_eval = batch

                loss = eval_model(inputs_eval, model, loss_func)

                list_loss_batch_eval.append(loss.item())

            list_loss_eval.append(sum(list_loss_batch_eval) / len(list_loss_batch_eval))

        print(
            f"epoch : {epoch} , loss_train : {list_loss[-1]} , loss_val : {list_loss_eval[-1]} "
        )


plt.plot(list_loss)


# In[ ]:


plt.plot(list_loss_eval)


print(min(list_loss_eval), min(list_loss))
