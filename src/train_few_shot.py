import torchvision
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import os
import torchvision.transforms.functional as TF
from tqdm import tqdm

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
parser.add_argument("--step_size", default=100, type=int)


args = parser.parse_args()


nb_ep = args.nb_ep
k = args.k
q = args.q
n = args.n
number_of_batch = args.nb_of_batch
step_size = args.step_size


class RotationTransform:
    """Rotate by one of the given angles."""

    def __init__(self, angles, fill):
        self.angles = angles
        self.fill = fill

    def __call__(self, x):
        angle = random.choice(self.angles)
        return TF.rotate(x, angle, fill=self.fill)


transform = torchvision.transforms.Compose(
    [
        RotationTransform(angles=[90, 180, 270], fill=255),
        torchvision.transforms.Resize(28, interpolation=2),
        torchvision.transforms.ToTensor(),
    ]
)

transform_eval = torchvision.transforms.Compose(
    [
        torchvision.transforms.Resize(28, interpolation=2),
        torchvision.transforms.ToTensor(),
    ]
)

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
    download=False,
    transform=transform,
    background=True,
)

eval_dataset = Omniglot(
    root="/staging/thesis_data_search/data",
    download=False,
    transform=transform_eval,
    background=False,
)


# ## Train

# In[12]:
def eval_model(inputs_eval, model, loss_func, accuracy=False):

    outputs = model(inputs_eval.to(device), nb_ep, n, k, q)
    targets = (
        torch.eye(k, device=device).unsqueeze(0).unsqueeze(-1).expand(nb_ep, k, k, q)
    )

    loss = loss_func(outputs, targets)

    if accuracy:
        accuracy = (outputs.argmax(dim=1) == targets.argmax(dim=1)).float().mean()

    return loss, accuracy


def train_model(inputs, model, loss_func, optim):

    model.zero_grad()

    loss, _ = eval_model(inputs, model, loss_func)

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

bg_taskloader = torch.utils.data.DataLoader(
    bg_dataset, batch_sampler=few_shot_sampler, num_workers=10
)


eval_taskloader = torch.utils.data.DataLoader(
    eval_dataset, batch_sampler=few_shot_sampler_val, num_workers=10
)


# In[26]:


loss_func = nn.MSELoss()
model = RelationNet(in_channels=1, out_channels=64, device=device, debug=True)
model.to(device)


lr = args.lr

optim = torch.optim.Adam(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=step_size, gamma=0.1)

list_loss = []
list_loss_eval = []
accuracy_eval = []
## train

epochs = args.nb_epochs
nb_eval = args.nb_eval
period_eval = max(epochs // nb_eval, 1)


for epoch in tqdm(range(epochs)):

    list_loss_batch = []

    model.train()

    for batch_idx, batch in enumerate(bg_taskloader):

        inputs, labels = batch

        loss = train_model(inputs, model, loss_func, optim)

        list_loss_batch.append(loss.item())

    list_loss.append(sum(list_loss_batch) / len(list_loss_batch))

    scheduler.step()

    if epoch % period_eval == 0:

        with torch.no_grad():
            model.eval()

            list_loss_batch_eval = []
            accuracy = 0

            for batch_idx, batch in enumerate(eval_taskloader):

                inputs_eval, labels_eval = batch

                loss, accuracy_batch = eval_model(
                    inputs_eval, model, loss_func, accuracy=True
                )

                list_loss_batch_eval.append(loss.item())
                accuracy += accuracy_batch

            accuracy = accuracy / len(eval_taskloader)
            accuracy_eval.append(accuracy)

            list_loss_eval.append(sum(list_loss_batch_eval) / len(list_loss_batch_eval))

        lr = "{:.2e}".format(scheduler.get_last_lr()[0])
        print(
            f"epoch : {epoch} , loss_train : {list_loss[-1]} , loss_val : {list_loss_eval[-1]} , accuracy_eval = {accuracy_eval[-1]} ,lr : {lr} "
        )

# In[ ]:


plt.plot(list_loss_eval)


min(list_loss_eval), min(list_loss), max(accuracy_eval)


few_shot_accuracy_sampler = FewShotSampler(
    bg_dataset,
    number_of_batch=100,
    episodes=nb_ep,
    sample_per_class=n,
    classes_per_ep=k,
    queries=q,
)

accuracy_taskloader = torch.utils.data.DataLoader(
    bg_dataset, batch_sampler=few_shot_accuracy_sampler, num_workers=10
)

accuracy = 0

for batch_idx, batch in enumerate(accuracy_taskloader):

    with torch.no_grad():
        model.eval()

        inputs, labels = batch

        outputs, accuracy_batch = eval_model(inputs, model, loss_func, accuracy=True)

        accuracy += accuracy_batch


print(accuracy / len(accuracy_taskloader))
