# data_process_clean.py

import os
import glob
import pickle
from pathlib import Path
from collections import defaultdict

import numpy as np
from tqdm import tqdm


# -----------------------------
# 基本超参数
# -----------------------------
T_TOTAL = 12          # 6 history + 6 future
V_MAX   = 120         # 每个样本最多 120 个 agent
C_FEAT  = 11          # [frame_id, object_id, object_type, x,y,z,len,w,h,heading,mask]


RAW_ROOT       = "data/raw"
SCENE_A_DIR    = os.path.join(RAW_ROOT, "scene_A")
SCENE_B_TXT    = os.path.join(RAW_ROOT, "scene_B", "scene_B.txt")

OUT_DIR        = "data/processed"
TEST_OUT_PKL   = os.path.join(OUT_DIR, "test.pkl")
USERS_OUT_DIR  = os.path.join(OUT_DIR, "users_clean")  # 新增：按 user 切分后的数据放这里


# -----------------------------
# 工具函数
# -----------------------------
def ensure_dir(path: str):
    Path(path).mkdir(parents=True, exist_ok=True)


def parse_scene_A() -> list[dict]:
    """
    解析 data/raw/scene_A 下所有 result_XXXX_Y_frame.txt
    返回：
        frames: list[dict]，长度 = 全局帧数
        frames[t] 是一个 dict: {object_id: (obj_type, x,y,z,len,w,h,heading)}
    """
    frames = []

    txt_files = sorted(glob.glob(os.path.join(SCENE_A_DIR, "*.txt")))
    if not txt_files:
        print(f"[WARN] No txt files found in {SCENE_A_DIR}")
        return frames

    print(f"Parsing {SCENE_A_DIR}: {len(txt_files)} files")

    for fname in tqdm(txt_files, desc="scene_A", unit="file"):
        frame_dict: dict[int, dict[int, tuple]] = defaultdict(dict)
        # frame_dict[local_frame_id][object_id] = (obj_type, x,y,z,len,w,h,heading)

        with open(fname, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) < 10:
                    continue
                lf   = int(parts[0])
                oid  = int(parts[1])
                otyp = int(parts[2])
                x    = float(parts[3])
                y    = float(parts[4])
                z    = float(parts[5])
                L    = float(parts[6])
                W    = float(parts[7])
                H    = float(parts[8])
                head = float(parts[9])

                frame_dict[lf][oid] = (otyp, x, y, z, L, W, H, head)

        for lf in sorted(frame_dict.keys()):
            frames.append(frame_dict[lf])

    print(f"[scene_A] total frames: {len(frames)}")
    return frames


def parse_scene_B() -> list[dict]:
    """
    解析 data/raw/scene_B/scene_B.txt
    返回与 parse_scene_A 相同格式的 frames 列表。
    """
    frames = []

    if not os.path.exists(SCENE_B_TXT):
        print(f"[WARN] scene_B.txt not found: {SCENE_B_TXT}")
        return frames

    print(f"Parsing {SCENE_B_TXT}")

    frame_dict: dict[int, dict[int, tuple]] = defaultdict(dict)

    with open(SCENE_B_TXT, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 10:
                continue
            fid  = int(parts[0])   # 这里是全局 frame id
            oid  = int(parts[1])
            otyp = int(parts[2])
            x    = float(parts[3])
            y    = float(parts[4])
            z    = float(parts[5])
            L    = float(parts[6])
            W    = float(parts[7])
            H    = float(parts[8])
            head = float(parts[9])

            frame_dict[fid][oid] = (otyp, x, y, z, L, W, H, head)

    for fid in sorted(frame_dict.keys()):
        frames.append(frame_dict[fid])

    print(f"[scene_B] total frames: {len(frames)}")
    return frames


def build_samples_from_frames(frames: list[dict]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    给定一个 scene 的 frames 列表，构建滑动窗口样本。
    返回:
        features:  (N, 11, T_TOTAL, V_MAX)
        adjacency: (N, V_MAX, V_MAX)
        mean_xy:   (N, 2)
    """
    if len(frames) < T_TOTAL:
        return (
            np.zeros((0, C_FEAT, T_TOTAL, V_MAX), dtype=np.float32),
            np.zeros((0, V_MAX, V_MAX), dtype=np.float32),
            np.zeros((0, 2), dtype=np.float32),
        )

    samples_feat = []
    samples_adj  = []
    samples_mean = []

    print(f"Building samples (T={T_TOTAL}, V_max={V_MAX})...")
    for start in tqdm(range(0, len(frames) - T_TOTAL + 1), desc="sliding", unit="win"):
        end = start + T_TOTAL
        win_frames = frames[start:end]

        # 1) 统计窗口内所有出现过的 object_id，最多取前 V_MAX 个
        obj_ids = set()
        for fr in win_frames:
            obj_ids.update(fr.keys())
        obj_ids = sorted(obj_ids)
        if len(obj_ids) > V_MAX:
            obj_ids = obj_ids[:V_MAX]

        id_to_idx = {oid: i for i, oid in enumerate(obj_ids)}
        V = len(obj_ids)
        if V == 0:
            continue

        # 2) 初始化 feature 与 mask
        feat = np.zeros((C_FEAT, T_TOTAL, V_MAX), dtype=np.float32)

        for t, fr in enumerate(win_frames):
            for oid, (otyp, x, y, z, L, W, H, head) in fr.items():
                if oid not in id_to_idx:
                    continue
                j = id_to_idx[oid]

                feat[0, t, j] = float(t)     # frame_id (local)
                feat[1, t, j] = float(oid)   # object_id
                feat[2, t, j] = float(otyp)  # object_type
                feat[3, t, j] = x
                feat[4, t, j] = y
                feat[5, t, j] = z
                feat[6, t, j] = L
                feat[7, t, j] = W
                feat[8, t, j] = H
                feat[9, t, j] = head
                feat[10, t, j] = 1.0        # mask

        # 3) adjacency：先用单位矩阵占位（只保留自连边）
        adj = np.eye(V_MAX, dtype=np.float32)

        # 4) mean_xy：窗口内所有有效点的平均
        mask = feat[10]  # (T,V_MAX)
        x_vals = feat[3] * mask
        y_vals = feat[4] * mask
        denom = mask.sum()
        if denom > 0:
            mean_x = x_vals.sum() / denom
            mean_y = y_vals.sum() / denom
        else:
            mean_x = 0.0
            mean_y = 0.0
        mean_xy = np.array([mean_x, mean_y], dtype=np.float32)

        samples_feat.append(feat)
        samples_adj.append(adj)
        samples_mean.append(mean_xy)

    if not samples_feat:
        return (
            np.zeros((0, C_FEAT, T_TOTAL, V_MAX), dtype=np.float32),
            np.zeros((0, V_MAX, V_MAX), dtype=np.float32),
            np.zeros((0, 2), dtype=np.float32),
        )

    features = np.stack(samples_feat, axis=0)
    adjacency = np.stack(samples_adj, axis=0)
    mean_xy = np.stack(samples_mean, axis=0)

    print(f"Built {features.shape[0]} samples, feature shape = {features.shape}")
    return features, adjacency, mean_xy


def main():
    # 创建输出目录
    ensure_dir(OUT_DIR)
    ensure_dir(USERS_OUT_DIR)

    # 1) 解析两个 scene
    frames_A = parse_scene_A()
    frames_B = parse_scene_B()

    # 2) 各自构建样本，再拼在一起
    feat_A, adj_A, mean_A = build_samples_from_frames(frames_A)
    feat_B, adj_B, mean_B = build_samples_from_frames(frames_B)

    features = np.concatenate([feat_A, feat_B], axis=0)
    adjacency = np.concatenate([adj_A, adj_B], axis=0)
    mean_xy = np.concatenate([mean_A, mean_B], axis=0)

    N = features.shape[0]
    print(f"\n=== TOTAL SAMPLES ===\nN = {N}")

    if N == 0:
        print("[WARN] No samples built, please check raw data / parser assumptions.")
        return

    # -----------------------------
    # 3) 先切 global test，再把剩余样本均分成 5 份作为 5 个用户
    # -----------------------------
    rng = np.random.RandomState(0)
    idx = np.arange(N)
    rng.shuffle(idx)

    # 先拿出 test（例如 10%）
    test_ratio = 0.1
    n_test = int(test_ratio * N)
    idx_test = idx[:n_test]
    idx_rest = idx[n_test:]

    print(f"n_test = {n_test}, n_rest = {len(idx_rest)}")

    # 保存 test（干净的 global test）
    test_feat = features[idx_test]
    test_adj  = adjacency[idx_test]
    test_mean = mean_xy[idx_test]

    with open(TEST_OUT_PKL, "wb") as f:
        pickle.dump([test_feat, test_adj, test_mean], f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"[Saved] {TEST_OUT_PKL}   ({test_feat.shape[0]} samples)")

    # 把剩余样本均分给 5 个 user
    num_users = 5
    n_rest = len(idx_rest)
    # 均匀分配，前 remainder 个多一个样本
    base = n_rest // num_users
    rem  = n_rest % num_users

    user_indices_list = []
    start = 0
    for u in range(num_users):
        size = base + (1 if u < rem else 0)
        end = start + size
        user_indices_list.append(idx_rest[start:end])
        start = end

    # 检查一下
    assert start == n_rest, "Split error: not all samples assigned to users."

    # 依次保存 user1..user5 的“干净全集”
    for u, u_idx in enumerate(user_indices_list, start=1):
        u_feat = features[u_idx]
        u_adj  = adjacency[u_idx]
        u_mean = mean_xy[u_idx]

        out_path = os.path.join(USERS_OUT_DIR, f"user{u}_clean_all.pkl")
        with open(out_path, "wb") as f:
            pickle.dump([u_feat, u_adj, u_mean], f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"[Saved] {out_path}  ({u_feat.shape[0]} samples)")


if __name__ == "__main__":
    main()