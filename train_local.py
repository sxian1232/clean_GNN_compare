import argparse
import json
import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from model import Model
from xin_feeder_baidu import Feeder   # 你现有的数据 Feeder
import numpy as np

##############################################################
#                  CONFIG (可自行修改)
##############################################################

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
history_frames = 6
future_frames = 6
batch_size = 32
epochs = 1
lr = 0.001
prox_mu_default = 0.0        # =0 就是 FedAvg ，>0 就是 FedProx

##############################################################
#            数据预处理（保持与 eval_only 一致）
##############################################################

def preprocess_batch(ori_data):
    """
    ori_data: (N, 11, T, V)
    输出:
      data: (N, 4, T, V)    # dx, dy, heading, mask
      ori_xy: (N, 2, T, V)  # x,y 原始坐标
      mask: (N, 1, T, V)
    """
    # 取 [x, y, heading, mask]  = index [3,4,9,10]
    feat_id = [3, 4, 9, 10]
    ori = ori_data[:, feat_id].float().to(device)      # (N,4,T,V)

    xy = ori_data[:, 3:5].float().to(device)           # (N,2,T,V)
    mask = ori_data[:, 10:11].float().to(device)       # (N,1,T,V)

    # 差分位移
    data = ori.clone()
    new_mask = (data[:, :2, 1:] != 0) * (data[:, :2, :-1] != 0)
    data[:, :2, 1:] = (data[:, :2, 1:] - data[:, :2, :-1]) * new_mask
    data[:, :2, 0] = 0

    return data, xy, mask


##############################################################
#                    Smooth L1 损失（WS 风格）
##############################################################

def compute_ws_loss(pred, gt, mask):
    """
    pred, gt: (N,2,T,V)
    mask: (N,1,T,V)

    输出：一个标量 loss，用于 backward
    """
    diff = torch.abs(pred - gt) * mask
    num = torch.sum(mask)
    loss = torch.sum(diff) / (num + 1e-9)
    return loss


##############################################################
#                       训练入口
##############################################################

def train_one_local_model(train_data_path, init_model_path,
                          out_model_path, prox_mu=0.0):
    """
    prox_mu = 0   → FedAvg
    prox_mu > 0   → FedProx
    """

    # --------------------- Load Data ---------------------
    feeder = Feeder(
        data_path=train_data_path,
        graph_args={'max_hop': 2, 'num_node': 120},
        train_val_test='all'
    )
    loader = DataLoader(feeder, batch_size=batch_size,
                        shuffle=True, drop_last=False, num_workers=2)

    # --------------------- Load Model ---------------------
    model = Model(
        in_channels=4,
        graph_args={'max_hop': 2, 'num_node': 120},
        edge_importance_weighting=True
    ).to(device)

    # 加载初始化模型
    ckpt = torch.load(init_model_path, map_location=device)
    state = ckpt["xin_graph_seq2seq_model"]
    model.load_state_dict(state)

    # FedProx 用
    global_params = {k: v.clone().detach() for k, v in model.state_dict().items()}

    optimizer = optim.Adam(model.parameters(), lr=lr)

    # --------------------- Training Loop ---------------------
    for ep in range(epochs):
        model.train()
        for ori_data, A, _ in loader:
            A = A.float().to(device)
            data, ori_xy, mask = preprocess_batch(ori_data)

            inp = data[:, :, :history_frames, :]       # (N,4,6,V)
            gt = ori_xy[:, :, history_frames:, :]      # (N,2,6,V)
            m  = mask[:, :, history_frames:, :]        # (N,1,6,V)

            optimizer.zero_grad()
            pred = model(
                pra_x=inp,
                pra_A=A,
                pra_pred_length=future_frames,
                pra_teacher_forcing_ratio=0,
                pra_teacher_location=None
            )  # (N,2,6,V)

            # WS loss
            loss = compute_ws_loss(pred, gt, m)

            # FedProx μ/2 · ||w - w_global||²
            if prox_mu > 0:
                prox_term = 0.0
                for name, p in model.named_parameters():
                    prox_term += torch.sum((p - global_params[name]) ** 2)
                loss += prox_mu / 2 * prox_term

            loss.backward()
            optimizer.step()

    # --------------------- Save ---------------------
    out = {"xin_graph_seq2seq_model": model.state_dict()}
    os.makedirs(os.path.dirname(out_model_path), exist_ok=True)
    torch.save(out, out_model_path)

    print(f"[OK] Saved local model → {out_model_path}")
    return out_model_path


##############################################################
#                   CLI / 命令行接口
##############################################################

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_data", required=True)
    ap.add_argument("--init_model", required=True)
    ap.add_argument("--out_model", required=True)
    ap.add_argument("--prox_mu", type=float, default=prox_mu_default)
    args = ap.parse_args()

    train_one_local_model(
        train_data_path=args.train_data,
        init_model_path=args.init_model,
        out_model_path=args.out_model,
        prox_mu=args.prox_mu
    )

