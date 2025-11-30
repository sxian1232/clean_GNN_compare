# generate_user_round_data.py
#
# 功能：
#   给定某个用户的 clean_all.pkl，
#   1）按指定噪声模式 + 强度注入噪声
#   2）按比例切成 train / val
#   3）保存到指定 round 目录下：
#        user{u}_r{R}_train.pkl
#        user{u}_r{R}_val.pkl
#
# 数据格式保持与原始一致：pickle.dump([feat, adj, mean_xy], f)

import argparse
import pickle
from pathlib import Path

import numpy as np


def ensure_dir(path: str):
    Path(path).mkdir(parents=True, exist_ok=True)


def add_noise(
    feat: np.ndarray,
    mode: str,
    sigma_xy: float,
    sigma_all: float,
    seed: int,
    x_idx: int = 3,
    y_idx: int = 4,
    mask_idx: int = 10,
) -> np.ndarray:
    """
    feat 形状： (N, C, T, V)
    mode:
      - "none": 不加噪声
      - "xy"  : 只对 x/y 通道加 N(0, sigma_xy)
      - "all" : 对所有通道加 N(0, sigma_all)，但保留 mask 通道不动
    """
    if mode == "none":
        return feat.copy()

    rng = np.random.default_rng(seed)
    noisy = feat.copy()

    if mode in ("xy", "all") and sigma_xy > 0:
        # x/y 通道（3,4），形状是 (N, T, V)
        noise_x = rng.normal(0.0, sigma_xy, size=noisy[:, x_idx].shape)
        noise_y = rng.normal(0.0, sigma_xy, size=noisy[:, y_idx].shape)
        noisy[:, x_idx] += noise_x
        noisy[:, y_idx] += noise_y

    if mode == "all" and sigma_all > 0:
        # 对所有通道加噪，但保留 mask 通道
        noise_all = rng.normal(0.0, sigma_all, size=noisy.shape)
        noise_all[:, mask_idx] = 0.0   # 不动 mask
        noisy += noise_all

    return noisy


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=str, required=True,
                    help="用户 clean_all.pkl 路径")
    ap.add_argument("--output_dir", type=str, required=True,
                    help="输出目录，例如 data/processed/users_rounds/round1")
    ap.add_argument("--user", type=int, required=True,
                    help="用户 ID，例如 1..5")
    ap.add_argument("--round", type=int, required=True,
                    help="第几轮，用于文件命名")
    ap.add_argument("--noise_mode", type=str, default="none",
                    choices=["none", "xy", "all"],
                    help="噪声模式：none / xy / all")
    ap.add_argument("--sigma_xy", type=float, default=0.0,
                    help="x/y 通道噪声强度")
    ap.add_argument("--sigma_all", type=float, default=0.0,
                    help="all 模式下，全通道噪声强度")
    ap.add_argument("--seed", type=int, default=0,
                    help="随机种子（同时用于噪声 & 划分 train/val）")
    ap.add_argument("--val_ratio", type=float, default=0.2,
                    help="验证集占比，0~1 之间")
    # 可选：允许自定义通道索引
    ap.add_argument("--x_idx", type=int, default=3)
    ap.add_argument("--y_idx", type=int, default=4)
    ap.add_argument("--mask_idx", type=int, default=10)

    args = ap.parse_args()

    # 基本检查
    if not (0.0 < args.val_ratio < 1.0):
        raise ValueError(f"val_ratio 必须在 (0,1) 内，目前是 {args.val_ratio}")

    # 读取 clean_all
    print(f"[INFO] Loading clean data from: {args.input}")
    with open(args.input, "rb") as f:
        feat, adj, mean_xy = pickle.load(f)

    print(f"[INFO] feat shape = {feat.shape}, adj shape = {adj.shape}, mean_xy shape = {mean_xy.shape}")

    # 注入噪声（只在 feature 上动手）
    print(f"[INFO] Applying noise: mode={args.noise_mode}, "
          f"sigma_xy={args.sigma_xy}, sigma_all={args.sigma_all}, seed={args.seed}")
    noisy_feat = add_noise(
        feat,
        mode=args.noise_mode,
        sigma_xy=args.sigma_xy,
        sigma_all=args.sigma_all,
        seed=args.seed,
        x_idx=args.x_idx,
        y_idx=args.y_idx,
        mask_idx=args.mask_idx,
    )

    N = noisy_feat.shape[0]
    indices = np.arange(N)
    rng = np.random.default_rng(args.seed)
    rng.shuffle(indices)

    n_val = int(N * args.val_ratio)
    n_train = N - n_val

    idx_train = indices[:n_train]
    idx_val = indices[n_train:]

    def subset(arr, idx):
        return arr[idx]

    feat_train = subset(noisy_feat, idx_train)
    adj_train = subset(adj, idx_train)
    mean_train = subset(mean_xy, idx_train)

    feat_val = subset(noisy_feat, idx_val)
    adj_val = subset(adj, idx_val)
    mean_val = subset(mean_xy, idx_val)

    # 输出目录 & 文件名
    out_dir = Path(args.output_dir)
    ensure_dir(out_dir)

    train_path = out_dir / f"user{args.user}_r{args.round}_train.pkl"
    val_path = out_dir / f"user{args.user}_r{args.round}_val.pkl"

    with open(train_path, "wb") as f:
        pickle.dump([feat_train, adj_train, mean_train], f, protocol=pickle.HIGHEST_PROTOCOL)
    with open(val_path, "wb") as f:
        pickle.dump([feat_val, adj_val, mean_val], f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"[OK] Saved train -> {train_path}  (N={feat_train.shape[0]})")
    print(f"[OK] Saved val   -> {val_path}    (N={feat_val.shape[0]})")


if __name__ == "__main__":
    main()
