# Learning to recognise rare traffic signs

This repo contains code for the paper : "Learning to recognise rare traffic signs"
enses. 

## How to use this code 

before doing anything be sure to either work in a virtual environement where the requirement.txt is install or in the docker image

### Download the dataset and extract the patches

* first donwload and unzip all of the training, validation and test set from the mapilarry website for the traffic sign dataset

* extract the patch

```shell
python src/patch_extraction.py --img_path=MAPILLARY_FOLDER/images --annot_path=MAPILLARY_FOLDER/annotations --output_path=YOUR__OUTPUT_FOLDER/patches
```

### run the script


```shell
python searching.py  --runs=10 --model=StandardNet  --dataset=2 --result_file=standard-net-searching.pkl 
```

```shell
python active_loop.py --ep=10 --top=50 --runs=1 --model=StandardNet --dataset=1 --result_file=standard-net-al.pkl
```


