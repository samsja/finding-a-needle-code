import numpy as np
import torch
import matplotlib.pyplot as plt

def img_from_tensor(inp):
    inp = inp.to("cpu").numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    
    return inp

def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = img_from_tensor(inp)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)


def plot_list(images,title=None,ncols=4,figsize=(8,8),img_from_tensor=img_from_tensor):
   


    if ncols > len(images):
        ncols = len(images)

    quotient = len(images) // ncols
    rest = len(images) % ncols
    
    fig = plt.figure(figsize=figsize)
    columns = ncols
    rows = quotient  if rest > 0 else 1

    for i in range(1, columns * rows + 1):
        fig.add_subplot(rows, columns, i)
        imshow(images[i - 1],title=title[i-1] if title is not None else None)
        plt.axis("off")
    plt.show()
