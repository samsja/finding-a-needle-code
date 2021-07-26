# Thesis Data Search

This repo contains code for the Thesis "Data Search"

NOTE :

This repo is using the imagenet and mapilarry dataset that are under certains licenses. 

The software in this repo is NOT production code, but only the code for the thesis. If needed to reimplement the few shot learning algorithm feel free to be inspired by the code in this repo but reimplement it in the silk framework of Zenseact.



## How to use this code 

before doing anything be sure to either work in a virtual environement where the requirement.txt is install or in the docker image

### Download the dataset and extract the patches

* first donwload and unzip all of the training, validation and test set from the mapilarry website for the traffic sign dataset

* extract the patch

´´´shell
python src/patch_extraction.py --img_path=MAPILLARY_FOLDER/images --annot_path=MAPILLARY_FOLDER/annotations --output_path=YOUR__OUTPUT_FOLDER/patches
´´´

### run the script


´´´shell
python python searching.py  --runs=1 --model=RelationNet  --dataset=2 --result_file=data/results/search_20_07_sy/relation-net_0.pkl"\nsbatch -J SRF10 exp.sh "python searching.py  --runs=10 --model=RelationNetFull  --dataset=2 --limit_search=50  --result_file=standard-net.pkl 
´´´

´´´shell
python active_loop.py --ep=10 --top=50 --runs=1 --model=StandardNet --dataset=1 --result_file=standard-net.pkl
´´´


