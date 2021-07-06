import torch

from src.utils_active_loop import exp_active_loop, get_data_6_rare, get_data_25_rare


import argparse


if __name__ == "__main__":


    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", default=1, type=int)
    parser.add_argument("--model",default="StandardNet",type=str)
    parser.add_argument("-N",default=1,type=int)
    parser.add_argument("--result_file",type=str)
    parser.add_argument("--limit_search",default=None,type=int)
    parser.add_argument("--path_data",default= "/staging/thesis_data_search/data",type=str)

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    batch_size = 256
    N = args.N

    number_of_runs = args.runs

    path_data = args.path_data

    class_to_search_on,init_data = get_data_25_rare(path_data,N,args.limit_search)


    results_df = exp_active_loop(
        N,
        class_to_search_on,
        episodes,
        number_of_runs,
        top_to_select,
        epochs_step,
        nb_of_eval,
        lr,
        device,
        init_data,
        batch_size,
        args.model,
        search=True,
        num_workers=8,
        retrain=True,
    )

    results_df.to_pickle(args.result_file)
