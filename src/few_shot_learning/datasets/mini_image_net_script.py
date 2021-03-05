import os
import shutil
from os.path import join

import argparse
import sys
from tqdm import tqdm

parser = argparse.ArgumentParser()

parser.add_argument("--path", type=str)
parser.add_argument("--output-path", type=str)
parser.add_argument("-f", type=str)

args = parser.parse_args()

path = args.path
output_path = args.output_path
file_source_image = args.f



with open(file_source_image) as file:

    folder = None
    
    for line in tqdm(file.readlines()):

        if line[0] in [" ","\n"]:
            continue
        elif line[-2] == ":":

            folder = line.split("/")[-2]

            try:
                os.makedirs(join(output_path, folder))
            except FileExistsError:
                pass

            continue

        img_name = line[:-1]

        try:
             shutil.copyfile(
                join(path, join(folder, img_name)),
                join(output_path, join(folder, img_name)),
            )
        except FileNotFoundError as e:
           
            print(
                f"file {join(path,join(folder,img_name))} does not exist",
                file=sys.stderr,
            )
