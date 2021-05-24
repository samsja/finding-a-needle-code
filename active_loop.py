import torch

from src.utils_active_loop import exp_active_loop, init_dataset


import argparse
if __name__ == "__main__":


    parser = argparse.ArgumentParser()
    parser.add_argument("--ep", default=10, type=int)
    parser.add_argument("--runs", default=1, type=int)
    parser.add_argument("--model",default="StandardNet",type=str)
    parser.add_argument("-N",default=1,type=int)
    parser.add_argument("--top",default=50,type=int)
    parser.add_argument("--result_file",type=str)
    parser.add_argument("--limit_search",default=None,type=int)


    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    path_data = "/staging/thesis_data_search/data"
    class_to_search_on = torch.Tensor([25, 26, 32, 95, 152, 175]).long()
    batch_size = 256




    support_filenames = [
        "/staging/thesis_data_search/data/patches/warning--slippery-road-surface--g1/1X0IkHf4hyDyWv9jdL25uXQ.jpg",
        "/staging/thesis_data_search/data/patches/warning--curve-left--g1/0vsCk34R8N_nq31Hjadwt4Q.jpg",
        "/staging/thesis_data_search/data/patches/regulatory--no-overtaking--g2/1eP65vCRiyu_x8nOJl_otsg.jpg",
        "/staging/thesis_data_search/data/patches/regulatory--no-stopping--g2/1x3iZWxvj6fTaLeBiiTPeEA.jpg",
        "/staging/thesis_data_search/data/patches/regulatory--maximum-speed-limit-20--g1/1t7pO54Ujrat7T33j3uGTOg.jpg",
        "/staging/thesis_data_search/data/patches/warning--slippery-road-surface--g2/1PYQrF98Be90rnFsFBpO6Qg.jpg",
    ]

    support_filenames = {key.item() : [support_filenames[i]] for i,key in enumerate(class_to_search_on) }


    N = args.N

    mask = class_to_search_on
    episodes = args.ep
    number_of_runs = args.runs
    top_to_select = args.top
    epochs_step = [20] * episodes
    lr = 1e-3

    init_data = lambda: init_dataset(path_data, class_to_search_on, support_filenames, N=N,limit_search=args.limit_search)

    scores_df = exp_active_loop(
        N,
        mask,
        episodes,
        number_of_runs,
        top_to_select,
        epochs_step,
        lr,
        device,
        init_data,
        batch_size,
        args.model,
        search=True,
        nb_of_eval=1
    )

    scores_df.to_pickle(args.result_file)
