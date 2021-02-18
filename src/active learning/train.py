import torch
import itertools 

from data import LabeledDataset, UnlabeledDataset
from torch.utils.data import DataLoader

from retrieve import retrieve


EPOCHS = range(100)
RUNS = range(10)

#model = ...

labeled_ds = LabeledDataset(path_, k)
labeled_dl = DataLoader(labeled_ds, batch_size = 10)

unlabeled_ds = UnlabeledDataset(labeled_ds.unlabeled)
unlabeled_dl = DataLoader(unlabeled_ds, batch_size = 10)

for run in RUNS:

    for epoch in EPOCHS:

        #Train


        #Eval


    #Retrieve
    retrieve(model, labeled_ds, unlabeled_dl, k)



