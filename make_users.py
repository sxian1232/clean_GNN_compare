# make_users.py
import pickle
import numpy as np
from pathlib import Path

IN_PATH = "data/processed/train.pkl"
OUT_DIR = Path("data/processed/users_clean")   # clean 数据专用目录
OUT_DIR.mkdir(parents=True, exist_ok=True)

NUM_USERS = 5
RANDOM_SEED = 42

def main():
    feat, adj, mean_xy = pickle.load(open(IN_PATH, "rb"))
    N = feat.shape[0]
    print(f"Global train samples: {N}")

    rng = np.random.RandomState(RANDOM_SEED)
    idx = np.arange(N)
    rng.shuffle(idx)

    splits = np.array_split(idx, NUM_USERS)

    for u, idx_u in enumerate(splits, start=1):
        idx_u = np.array(idx_u, dtype=np.int64)
        f_u = feat[idx_u]
        a_u = adj[idx_u]
        m_u = mean_xy[idx_u]

        out_path = OUT_DIR / f"user{u}_clean_train.pkl"
        with open(out_path, "wb") as f:
            pickle.dump([f_u, a_u, m_u], f, protocol=pickle.HIGHEST_PROTOCOL)

        print(f"[CLEAN] User{u}: {len(idx_u)} samples → {out_path}")

    print("\nDone. 5 clean users generated.")

if __name__ == "__main__":
    main()