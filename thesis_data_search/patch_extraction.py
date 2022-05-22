import json
import numpy as np
from PIL import Image
import os
import sys
import click
from tqdm.auto import tqdm


@click.command()
@click.option("--img_path", required=True, help="the path to the image folder")
@click.option("--annot_path", required=True, help="the path to the annotation folder")
@click.option("--output_path", required=True, help="the path to the output folder")
def main(img_path, annot_path, output_path):

    metadata_path = annot_path

    output_path_images = f"{output_path}/images"
    output_path_annotations = f"{output_path}/annotations"

    list_file = os.listdir(annot_path)

    for json_filename in tqdm(list_file):

        if "json" not in json_filename:
            continue

        file_id = json_filename[:-5]
        image_filename = file_id + ".jpg"

        with open(f"{annot_path}/{json_filename}") as json_file:
            data = json.load(json_file)

        try:
            img = np.array(Image.open(f"{img_path}/{image_filename}"))
        except:
            print(f"{image_filename} can't be loaded as an image", file=sys.stderr)
            continue

        signs_ = data["objects"]

        for sign in signs_:
            if "NotListed" not in sign["label"]:

                class_ = sign["label"]

                patch_name = sign["key"]

                x1 = int(sign["bbox"]["xmin"])
                x2 = int(sign["bbox"]["xmax"])

                y1 = int(sign["bbox"]["ymin"])
                y2 = int(sign["bbox"]["ymax"])

                patch = img[y1:y2, x1:x2]

                if not os.path.isdir(f"{output_path_images}/{class_}"):
                    os.makedirs(f"{output_path_images}/{class_}")

                Image.fromarray(patch).save(
                    f"{output_path_images}/{class_}/{patch_name}.jpg"
                )

                if not os.path.isdir(f"{output_path_annotations}/{class_}"):
                    os.makedirs(f"{output_path_annotations}/{class_}")

                with open(
                    f"{output_path_annotations}/{class_}/{patch_name}.json", "w"
                ) as outfile:
                    json.dump(sign, outfile)


if __name__ == "__main__":
    main()
