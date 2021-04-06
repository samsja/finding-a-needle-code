import json
import numpy as np
from PIL import Image
import os
import sys

import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--data_path", default= "data/traffic-signs", type=str)
parser.add_argument("--output_path", default= "data/traffic-signs/patched", type=str)



args = parser.parse_args()


data_path = args.data_path
output_path = args.output_path

sub_path = "converted"
img_path = f"{data_path}/{sub_path}/images/"
annot_path = f"{data_path}/{sub_path}/annotations/"
metadata_path = f"{data_path}/original/metadata/metadata"

output_path_images = f"{output_path}/images"
output_path_annotations = f"{output_path}/annotations"




for json_filename in os.listdir(annot_path):

    if "json" not in json_filename:
        continue

    file_id = json_filename[:-5]

    with open(annot_path + json_filename) as json_file:
        data = json.load(json_file)
    image_filename = data["Scenes"]["1"]["Sensors"]["FC"]["Filename"]

    try:
        img = np.array(Image.open(img_path + image_filename))
    except:
        print(f"{image_filename} can't be loaded as an image", file=sys.stderr)
        continue

    with open(f"{metadata_path}/{file_id}.json") as jsom_metadata_file:
        metadata = json.load(jsom_metadata_file)

    annotations = metadata

    signs_ = data["Scenes"]["1"]["TrafficSigns"]

    for o in signs_:
        if "NotListed" not in signs_[o]["2dMarking"]["FC"]["SignProperties"]["Type"]:

            class_ = signs_[o]["2dMarking"]["FC"]["SignProperties"]["Type"]

            x1 = int(signs_[o]["2dMarking"]["FC"]["Top"]["X"])
            x2 = int(signs_[o]["2dMarking"]["FC"]["Bottom"]["X"])
         
            assert(x1<x2)

            x_diff = (x2 - x1)/2
            x1 -= x_diff/2
            x1 = int(max(0,x1))
            
            x2 += x_diff/2
            x2 = int(min(x2,img.shape[1] -1))

            y1 = int(signs_[o]["2dMarking"]["FC"]["Top"]["Y"])
            y2 = int(signs_[o]["2dMarking"]["FC"]["Bottom"]["Y"])

            assert(y1<y2)

            y_diff = (y2 - y1)/2
            y1 -= y_diff/2
            y1 = int(max(0,y1))
            
            y2 += y_diff/2
            y2 = int(min(y2,img.shape[0] -1 ))


            patch = img[y1:y2, x1:x2]

            if not os.path.isdir(f"{output_path_images}/{class_}"):
                os.makedirs(f"{output_path_images}/{class_}")   

            Image.fromarray(patch).save(f"{output_path_images}/{class_}/{o}.jpg")

            if not os.path.isdir(f"{output_path_annotations}/{class_}"):
                os.makedirs(f"{output_path_annotations}/{class_}")

            with open(f"{output_path_annotations}/{class_}/{o}.json", "w") as outfile:
                json.dump(annotations, outfile)
