#!/usr/bin/env python
# -*- coding: utf-8 -*-


from bokeh.io import curdoc
import pandas as pd

import umap
import umap.plot

import colorcet as cc
import datashader as ds
import datashader.transfer_functions as tf

import torch

from tqdm.auto import tqdm
import numpy as np

def get_features_and_label(dataloader,feature_extractor,device):
    features = []
    labels = []

    with torch.no_grad():

        """
        for class_ in tqdm(classes):
            sampler = ClassSampler(train_dataset, class_, batch_size)

            loader = torch.utils.data.DataLoader(
                train_dataset, batch_sampler=sampler, num_workers=20
        )
        """

        for idx, batch in enumerate(tqdm(dataloader, disable=False)):

            batch_features = feature_extractor(batch["img"].to(device))
            features.append(batch_features)
            labels.append(batch["label"].to(device))

        features = torch.cat(features)
        labels = torch.cat(labels)

        features_np = features.to("cpu").numpy()
        labels_np = labels.to("cpu").numpy()

    return features_np, labels_np


def prepare_to_plot(umap_2d,labels,labels_test):

    df = pd.DataFrame(umap_2d, columns=("x", "y"))
    df["class"] = pd.Series([str(x) for x in labels], dtype="category")
    df["test"] = pd.Series([str(x in labels_test) for x in labels], dtype="category")

    return df

def plot_two_sided(df):
    cvs = ds.Canvas(plot_width=700, plot_height=700)

    agg = cvs.points(df, "x", "y", ds.count_cat("class"))
    agg_test = cvs.points(df, "x", "y", ds.count_cat("test"))

    img_split =    tf.shade(
            agg_test,
            cmap=["red", "blue"],
            how="eq_hist",
            name=" Traffic sign cnn output into two dimensions by UMAP split test/train",
        )

    img_full =     tf.shade(
            agg,
            color_key=cc.glasbey + cc.glasbey_cool,
            how="eq_hist",
            name=" Traffic sign cnn output into two dimensions by UMAP ",
        )

    img = tf.Images(
        img_split,img_full
    )

    return img


def plot_bokeh(df,reducer,labels):
    curdoc().clear()
    p = umap.plot.interactive(reducer, labels=labels, hover_data=df, point_size=2)
    umap.plot.output_notebook()
    umap.plot.show(p)


def get_features_labels_on_rare_class(dataset,class_to_search_on,feature_extractor,device) :
    with torch.no_grad():
        features_val = []
        labels_val = []
        for class_ in class_to_search_on:
            for idx in dataset.get_index_in_class(class_):
                features_val.append(feature_extractor(dataset[idx]["img"].to(device).unsqueeze(dim=0)))
                labels_val.append(dataset[idx]["label"].item())

        features_val = torch.cat(features_val)
        labels_val = np.array(labels_val)
    
    return features_val,labels_val

def create_df(umap_data, labels):
    df = pd.DataFrame(umap_data, columns=("x", "y"))
    df["class"] = pd.Series(labels)
    return df
