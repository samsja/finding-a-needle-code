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


def plot_list(images,title=None,ncols=4,figsize=(7,7),img_from_tensor=img_from_tensor):
    quotient = images.shape[0] // ncols
    rest = images.shape[0] % ncols
    
    nrows = quotient + 1 if rest > 0 else 1
    
    fig,ax = plt.subplots(nrows=nrows,ncols=ncols, constrained_layout=True,figsize=figsize)
    
    if title is not None:
        if len(title) != len(images):
            raise ValueError
        
    for i in range(nrows):
        for j in range(ncols):
            ax[i,j].axis("off")
                 
    for i,image in enumerate(images):
        (x,y) = (i//ncols,i%ncols)
        ax[x,y].imshow(img_from_tensor(image))
        
        if title is not None:
            ax[x,y].set_title(title[i])

