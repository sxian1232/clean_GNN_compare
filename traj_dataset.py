# traj_dataset.py
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset

from layers.graph import Graph   # 使用原来的 Graph 工具


class TrajDataset(Dataset):
    """
    Clean Feeder:

    输入：train.pkl / val.pkl / test.pkl
        [features, adjacency, mean_xy]

    输出：
        feat: (11, T, V)
        A:    (K, V, V)
        mean: (2,)
    """

    def __init__(self, data_path, graph_args=None):
        super().__init__()

        graph_args = graph_args or {'max_hop': 2, 'num_node': 120}

        with open(data_path, "rb") as f:
            data = pickle.load(f)

        self.features = data[0]   # (N, 11, T, V)
        self.adj_raw = data[1]    # (N, V, V)
        self.mean_xy = data[2]    # (N, 2)
        self.N = len(self.features)

        self.graph = Graph(**graph_args)

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        feat = self.features[idx]          # (11, T, V)
        adj  = self.adj_raw[idx]           # (V, V)
        mean = self.mean_xy[idx]           # (2,)

        # Graph: K 路 adjacency
        A = self.graph.get_adjacency(adj)
        A = self.graph.normalize_adjacency(A)
        A = A.astype(np.float32)            # ⭐ 强制 float32

        return feat.astype(np.float32), A, mean.astype(np.float32)