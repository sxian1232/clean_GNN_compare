# make_initial_global_dataset.py
#
# 将 user1~user5 的干净 all.pkl 合并成一个 initial global train.pkl
# 用于训练 initial clean global model（不需要 val / test）
#
# 输出：
#   data/processed/initial_global/train.pkl

import pickle
import numpy as np
from pathlib import Path

USER_DIR = Path("data/processed/users_clean")
OUT_DIR = Path("data/processed/initial_global")
OUT_DIR.mkdir(parents=True, exist_ok=True)

def main():
    feats = []
    adjs = []
    means = []

    print("=== Building Initial Global Dataset ===")

    for u in range(1, 6):
        path = USER_DIR / f"user{u}_clean_all.pkl"
        print(f"Loading {path}")

        feat_u, adj_u, mean_u = pickle.load(open(path, "rb"))
        feats.append(feat_u)
        adjs.append(adj_u)
        means.append(mean_u)

    # concatenate
    feat_all = np.concatenate(feats, axis=0)
    adj_all = np.concatenate(adjs, axis=0)
    mean_all = np.concatenate(means, axis=0)

    print(f"Total samples = {feat_all.shape[0]}")

    # shuffle for training
    idx = np.arange(feat_all.shape[0])
    np.random.shuffle(idx)

    feat_all = feat_all[idx]
    adj_all = adj_all[idx]
    mean_all = mean_all[idx]

    out_path = OUT_DIR / "train.pkl"
    pickle.dump([feat_all, adj_all, mean_all], open(out_path, "wb"))

    print(f"[OK] Saved → {out_path}")

if __name__ == "__main__":
    main()
