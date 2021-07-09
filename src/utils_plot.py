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
    plt.axis("off")
    if title is not None:
        plt.title(title)


def plot_list(images,title=None,ncols=4,figsize=(8,8),img_from_tensor=img_from_tensor):
   


    if ncols > len(images):
        ncols = len(images)

    quotient = len(images) // ncols
    rest = len(images) % ncols
    
    fig = plt.figure(figsize=figsize)
    columns = ncols
    rows = quotient + 1  if rest > 0 else 1
    for i in range(1, len(images)+1):
        fig.add_subplot(rows, columns, i)
        imshow(images[i - 1],title=title[i-1] if title is not None else None)
        plt.axis("off")
    plt.show()



def plot_search(
    n,
    top,
    relation,
    test_dataset,
    figsize=(7, 7),
    img_from_tensor=img_from_tensor,
    ncols=4,
):
    top_n = torch.stack([test_dataset[i.item()]["img"] for i in top[0:n]])
    title = ["{:.2f}".format(r.item()) for r in relation[0:n]]

    plot_list(
        top_n,
        title=title,
        figsize=figsize,
        img_from_tensor=img_from_tensor,
        ncols=ncols,
    )


def plot_image_to_find(class_to_search_for, test_dataset, relation, top, max_len=10):

    index_to_find = test_dataset.get_index_in_class(class_to_search_for)
    max_len = min(max_len, len(index_to_find) - 1)

    index_to_find = index_to_find[:max_len]

    top = top.squeeze(1)
    relation = torch.stack([relation[top == i] for i in index_to_find])

    title = ["{:.2f}".format(r.item()) for r in relation]

    target_images = torch.stack([test_dataset[i]["img"] for i in index_to_find])

    plot_list(target_images, title=title,ncols=6,figsize=(15,15))



def plot_score_one_class(score, class_, scores_df):
    plt.plot(list(scores_df[scores_df["class"] == class_][score]), label=f"{class_}")


def plot_mean(score, scores_df):
    plt.plot(
        list(scores_df[["iteration", score]].groupby(["iteration"]).mean()[score]),
        label="mean",
        linewidth=4,
    )


def plot_score(score, scores_df):
    for class_ in scores_df["class"].unique():
        plot_score_one_class(score, class_, scores_df)

    plot_mean(score, scores_df)
    plt.title(f"{score}")
    plt.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.05),
        fancybox=True,
        shadow=True,
        ncol=5,
    )


def plot_score_model(score, model, scores_df):
    plot_score(score, scores_df[scores_df["model"] == model])


def plot_all_model_mean(score, scores_df):
    for model in scores_df["model"].unique():
        df = scores_df[scores_df["model"] == model]
        plt.plot(
            list(df[["iteration", score]].groupby(["iteration"]).mean()[score]),
            label=model,
        )
    plt.title(f"{score}")
    plt.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.05),
        fancybox=True,
        shadow=True,
        ncol=5,
    )




