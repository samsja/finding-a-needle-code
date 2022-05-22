# Improving traffic sign recognition by active search

This repo contains code for the paper : [Improving traffic sign recognition by active search](https://arxiv.org/abs/2111.14426)

## How to use this code 

before doing anything be sure to either work in a virtual environement where the requirement.txt is install or in the docker image

### Download the dataset and extract the patches

* first donwload and unzip all of the training, validation and test set from the mapilarry website for the traffic sign dataset

* extract the patch

```shell
python thesis_data_search/patch_extraction.py --img_path=MAPILLARY_FOLDER/images --annot_path=MAPILLARY_FOLDER/annotations --output_path=YOUR__OUTPUT_FOLDER/patches
```

### run the script


* For the active loop on the 25 rarest traffic signs classes
```shell
python active_loop.py --top=50 --ep=5  --model=StandardNet --result_file=standard-net-al.pkl --path_data=path_to_data
```

* For the active loop on synthetic data:
```shell
python active_loop.py --top=50 --ep=5  --limit_search=50 --model=StandardNet --result_file=non-synthetic-al.pkl --dataset=0
python active_loop.py --top=50 --ep=5  --limit_search=50 --model=StandardNet --result_file=synthetic-al.pkl --dataset=2
```

* For the active loop with few shot learning:
```shell
python active_loop.py --top=50 --ep=5  --model=ProtoNetFull --result_file=proto-net-al.pkl
```

* For the searching only:
```shell
python searching.py  --model=StandardNet  --result_file=standard-net-searching.pkl 
```


