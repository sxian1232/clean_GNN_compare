# eval_model.py
import argparse
import json
import torch
import numpy as np
from torch.utils.data import DataLoader

from traj_dataset import TrajDataset
from model import Model


def eval_ws(model, loader, device):
    model.eval()
    total_sum = np.zeros(6, dtype=np.float64)
    total_count = np.zeros(6, dtype=np.float64)

    with torch.no_grad():
        for feat, A, mean_xy in loader:
            feat = feat.to(device)
            A = A.to(device)

            # history = 6, future = 6
            input_data = feat[:, :, :6, :]
            output_gt = feat[:, :2, 6:, :]
            output_mask = feat[:, 10:, 6:, :]

            pred = model(
                pra_x=input_data,
                pra_A=A,
                pra_pred_length=6,
                pra_teacher_forcing_ratio=0,
                pra_teacher_location=None
            )

            diff = (pred - output_gt).abs() * output_mask
            sum_t = diff.sum(dim=1).sum(dim=-1).cpu().numpy()
            cnt_t = output_mask.sum(dim=1).sum(dim=-1).cpu().numpy()

            total_sum += sum_t.sum(axis=0)
            total_count += cnt_t.sum(axis=0)

    ws = total_sum / (total_count + 1e-9)
    return list(ws), float(ws.sum())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--out_json", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    print("Loading model:", args.model)

    # Dataset
    ds = TrajDataset(args.data)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False)

    # Model architecture MUST be identical to train_local.py
    model = Model(
        in_channels=11,
        graph_args={"max_hop": 2, "num_node": 120},
        edge_importance_weighting=True
    ).to(device)

    # Load checkpoint correctly
    ckpt = torch.load(args.model, map_location=device)

    # Your saved format is {"xin_graph_seq2seq_model": model_state}
    if "xin_graph_seq2seq_model" in ckpt:
        model.load_state_dict(ckpt["xin_graph_seq2seq_model"])
    else:
        raise RuntimeError("Checkpoint does not contain key 'xin_graph_seq2seq_model'.")

    # Evaluate
    ws_per_h, ws_sum = eval_ws(model, dl, device)

    metrics = {
        "model": args.model,
        "data": args.data,
        "ws_per_horizon": ws_per_h,
        "ws_sum": ws_sum
    }

    with open(args.out_json, "w") as f:
        json.dump(metrics, f, indent=2)

    print("Saved metrics →", args.out_json)


if __name__ == "__main__":
    main()