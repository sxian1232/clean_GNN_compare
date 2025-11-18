# train_local.py
import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.optim as optim

from traj_dataset import TrajDataset
from model import Model


# -------------------------
#   Utils
# -------------------------
def ensure_dir(path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def eval_ws(model, loader, device):
    """
    返回：
        ws_per_horizon: list[6]
        ws_sum: float
    """
    model.eval()
    total_sum = np.zeros(6, dtype=np.float64)
    total_count = np.zeros(6, dtype=np.float64)

    with torch.no_grad():
        for feat, A, mean_xy in loader:
            feat = feat.to(device)
            A = A.to(device)

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


# -------------------------
#   Training
# -------------------------
def train_one_epoch(model, loader, optimizer, device):
    model.train()

    for i, (feat, A, _) in enumerate(loader):
        feat = feat.to(device)
        A = A.to(device)

        input_data = feat[:, :, :6, :]
        output_gt = feat[:, :2, 6:, :]
        output_mask = feat[:, 10:, 6:, :]

        pred = model(
            pra_x=input_data,
            pra_A=A,
            pra_pred_length=6,
            pra_teacher_forcing_ratio=0,
            pra_teacher_location=output_gt
        )

        diff = (pred - output_gt).abs() * output_mask
        loss = diff.sum() / (output_mask.sum() + 1e-9)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 10 == 0:
            print(f"Iter {i:04d} | loss {loss.item():.6f}")


# -------------------------
#   Main
# -------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data", type=str, required=True)
    parser.add_argument("--init_model", type=str, default="none")
    parser.add_argument("--out_model", type=str, required=True)
    parser.add_argument("--metrics_out", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=1)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # Dataset
    ds_train = TrajDataset(args.train_data)
    dl_train = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True)

    # Model
    model = Model(
        in_channels=11,         # ← 正确：feature 有 11 个通道
        graph_args={"max_hop": 2, "num_node": 120},
        edge_importance_weighting=True
    ).to(device)

    # -------- FIXED LOGIC HERE --------
    if args.init_model.lower() != "none":
        print(f"Loading init model: {args.init_model}")
        ckpt = torch.load(args.init_model, map_location="cpu")
        model.load_state_dict(ckpt['xin_graph_seq2seq_model'])
    else:
        print("No init model provided. Training from scratch.")
    # ----------------------------------

    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    for ep in range(args.epochs):
        print(f"\n===== Epoch {ep+1}/{args.epochs} =====")
        train_one_epoch(model, dl_train, optimizer, device)

    # Save model
    ensure_dir(args.out_model)
    torch.save({"xin_graph_seq2seq_model": model.state_dict()}, args.out_model)
    print("Saved model →", args.out_model)

    # Compute WS
    ws_per_h, ws_sum = eval_ws(model, dl_train, device)

    metrics = {
        "train_data": args.train_data,
        "init_model": args.init_model,
        "epochs": args.epochs,
        "ws_per_horizon": ws_per_h,
        "ws_sum": ws_sum
    }

    ensure_dir(args.metrics_out)
    with open(args.metrics_out, "w") as f:
        json.dump(metrics, f, indent=2)

    print("Saved metrics →", args.metrics_out)


if __name__ == "__main__":
    main()