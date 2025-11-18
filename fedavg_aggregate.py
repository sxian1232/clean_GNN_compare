# fedavg_aggregate.py
import os
import argparse
import torch

from model import Model


def load_user_weights(path, device):
    ckpt = torch.load(path, map_location=device)

    if "xin_graph_seq2seq_model" not in ckpt:
        raise RuntimeError(f"Model at {path} missing key 'xin_graph_seq2seq_model'")

    return ckpt["xin_graph_seq2seq_model"]


def average_state_dicts(dict_list):
    """FedAvg = element-wise average"""
    avg = {}

    # 初始化所有 key
    for k in dict_list[0].keys():
        avg[k] = sum(d[k] for d in dict_list) / len(dict_list)

    return avg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_paths", nargs="+", required=True,
                        help="List of user local model paths (user1.pt ... user5.pt)")
    parser.add_argument("--out_model", type=str, required=True,
                        help="Where to save aggregated global model")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    print("Aggregating models:", args.model_paths)

    # 1. Load all user weights
    state_dicts = []
    for p in args.model_paths:
        sd = load_user_weights(p, device)
        state_dicts.append(sd)

    print(f"Loaded {len(state_dicts)} user models.")

    # 2. Average
    global_sd = average_state_dicts(state_dicts)

    # 3. Build model arch (must match train)
    model = Model(
        in_channels=11,
        graph_args={"max_hop": 2, "num_node": 120},
        edge_importance_weighting=True
    ).to(device)

    # 4. Load averaged weights
    model.load_state_dict(global_sd)

    # 5. Save
    os.makedirs(os.path.dirname(args.out_model), exist_ok=True)
    torch.save({"xin_graph_seq2seq_model": model.state_dict()}, args.out_model)

    print("Saved global model →", args.out_model)
    print("FedAvg aggregation complete.")


if __name__ == "__main__":
    main()