import json
import numpy as np
from PIL import Image
import os




img_path = "traffic-signs/images/"
annot_path = "traffic-signs/annotations/"


for json_ in os.listdir(annot_path)[59:]:
    
    if "json" not in json_:
        continue
    
    with open(annot_path+json_) as json_file:
        data = json.load(json_file)

        fn = data["Scenes"]["1"]["Sensors"]["FC"]["Filename"]

        try:
            img = np.array(Image.open(img_path+fn))
        except:
            continue

        signs_ = data["Scenes"]["1"]["TrafficSigns"]
      

        for o in signs_:
            if "NotListed" not in signs_[o]["2dMarking"]["FC"]["SignProperties"]["Type"]:
                
                class_ = signs_[o]["2dMarking"]["FC"]["SignProperties"]["Type"]
                
                x1 = int(signs_[o]["2dMarking"]["FC"]["Top"]["X"])
                x2 = int(signs_[o]["2dMarking"]["FC"]["Bottom"]["X"])

                y1 = int(signs_[o]["2dMarking"]["FC"]["Top"]["Y"])
                y2 = int(signs_[o]["2dMarking"]["FC"]["Bottom"]["Y"])

                patch = img[y1:y2, x1:x2]
                
                
                if not os.path.isdir(class_):
                    os.mkdir(class_)

                Image.fromarray(patch).save(class_+"/"+o+".jpg")
      
