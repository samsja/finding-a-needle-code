import torch

from thesis_data_search.utils_search import exp_searching

from thesis_data_search.datasource import get_data_6_rare, get_data_25_rare, get_data_6_rare_sy

import argparse


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", default=1, type=int)
    parser.add_argument("--model", default="StandardNet", type=str)
    parser.add_argument("-N", default=1, type=int)
    parser.add_argument("--result_file", type=str)
    parser.add_argument("--limit_search", default=None, type=int)
    parser.add_argument(
        "--path_data", default="/staging/thesis_data_search/data", type=str
    )
    parser.add_argument("--dataset", default=0, type=int)

    args = parser.parse_args()

    print(args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    batch_size = 256
    N = args.N

    path_data = args.path_data

    datasc = [get_data_6_rare, get_data_25_rare, get_data_6_rare_sy]
    class_to_search_on, init_data = datasc[args.dataset](
        path_data, N, args.limit_search
    )

    scores_df = exp_searching(
        N,
        class_to_search_on,
        number_of_runs=args.runs,
        top_to_select=[5, 15, 50, 100, 1000],
        device=device,
        init_dataset=init_data,
        batch_size=batch_size,
        model_adapter_search_param=args.model,
        num_workers=8,
    )

    scores_df.to_pickle(args.result_file)
