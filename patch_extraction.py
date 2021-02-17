import json
import numpy as np
from PIL import Image
import os


sub_path = "converted"

data_path="traffic-signs"

img_path = f"{data_path}/{sub_path}/images/"
annot_path = f"{data_path}/{sub_path}/annotations/"
output_path = f"{data_path}/patched"

for json_ in os.listdir(annot_path):

    if "json" not in json_:
        continue
    with open(annot_path + json_) as json_file:
        data = json.load(json_file)

        fn = data["Scenes"]["1"]["Sensors"]["FC"]["Filename"]

        try:
            img = np.array(Image.open(img_path + fn))
        except:
            continue

        signs_ = data["Scenes"]["1"]["TrafficSigns"]

        for o in signs_:
            if (
                "NotListed"
                not in signs_[o]["2dMarking"]["FC"]["SignProperties"]["Type"]
            ):

                class_ = signs_[o]["2dMarking"]["FC"]["SignProperties"]["Type"]

                x1 = int(signs_[o]["2dMarking"]["FC"]["Top"]["X"])
                x2 = int(signs_[o]["2dMarking"]["FC"]["Bottom"]["X"])

                y1 = int(signs_[o]["2dMarking"]["FC"]["Top"]["Y"])
                y2 = int(signs_[o]["2dMarking"]["FC"]["Bottom"]["Y"])

                patch = img[y1:y2, x1:x2]

                if not os.path.isdir(f"{output_path}/{class_}"):
                    os.makedirs(f"{output_path}/{class_}")

                Image.fromarray(patch).save(f"{output_path}/{class_}/{o}.jpg")
