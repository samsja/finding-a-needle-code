import torch
import itertools

def retrieve(model: torch.module, labeled_ds: torch.DataSet, unlabeled_dl: torch.DataLoader, k: int, class_name: str):
    support_set = labeled_ds.get_support(class_name)
    support_set = model(support_set)
    support_set = support_set.mean() # Get mean feature vector for wanted class
    
    data_list = []

    for q in unlabeled_dl:
        img, fn, c = q["image"], q["fn"], q["label"]

        with torch.no_grad():
            probs = model.compute_prob(img, support_set)

        data_list.append((fn, c, probs))

    data_list = list(itertools.chain.from_iterable(data_list))
    sorted(data_list, lambda x : x[2])

    top_k = data_list[:k]

    labeled_ds.add_datapoints(top_k)
    unlabeled_dl.remove_dapoints(top_k)


    


