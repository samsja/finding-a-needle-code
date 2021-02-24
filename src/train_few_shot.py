import torchvision
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from tqdm import tqdm
import os



from few_shot_learning import FewShotDataSet, FewShotSampler,Omniglot,RelationNet

bg_dataset = Omniglot(
    root="./data", download=True, transform=torchvision.transforms.ToTensor(),background=True
)

eval_dataset = Omniglot(
    root="./data", download=True, transform=torchvision.transforms.ToTensor(),background=False
)


few_shot_sampler = FewShotSampler(bg_dataset,episodes=5,sample_per_class=1,classes_per_ep=5,queries=5)


nb_ep = few_shot_sampler.episodes
k = few_shot_sampler.classes_per_ep
n = few_shot_sampler.sample_per_class
q = few_shot_sampler.queries



epochs = 5
loss_func = nn.MSELoss()
model = RelationNet(in_channels=1,out_channels=64,debug=True)

lr = 1e-1

optim = torch.optim.Adam(model.parameters(),lr=lr)

list_loss = []

for epoch in tqdm(range(epochs)):
    
    bg_taskloader = torch.utils.data.DataLoader(
    bg_dataset,
    batch_sampler= few_shot_sampler,
    num_workers=1
    )
    
    for batch_idx,batch in enumerate(bg_taskloader):

        inputs,labels = batch

        inputs = inputs.view(k,n+q,1,105,105)
        labels = labels.view(k,n+q)

        support_inputs,support_labels = inputs[:,:n] , labels[:,:n] 
        queries_inputs, queries_labels = inputs[:,-q:] ,labels[:,-q:]
        
        targets = torch.eye(len(queries_inputs)).repeat(queries_inputs.size(1),1,1)

        
        outputs = model(support_inputs,queries_inputs)
        
        loss = loss_func(outputs,targets)
        
        model.zero_grad()

        loss.backward()

        list_loss.append(loss.item())
        
        optim.step()


plt.plot(list_loss)
