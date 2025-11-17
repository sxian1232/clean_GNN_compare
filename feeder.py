import pickle
import numpy as np
import torch
from torch.utils.data import Dataset

from layers.graph import Graph


class TrajDataset(Dataset):
    """
    Clean Feeder

    输入的 pkl 格式：
        features : (N, 11, T, 120), float
        adjacency: (N, 120, 120)
        mean_xy  : (N, 2)

    输出：
        feat  : (11, T, 120)  float32
        A     : (K, 120, 120) float32
        mean  : (2,)          float32
    """

    def __init__(self, data_path, graph_args=None):
        super().__init__()

        graph_args = graph_args or {'max_hop': 2, 'num_node': 120}

        # -------------------
        # 加载 pkl
        # -------------------
        with open(data_path, "rb") as f:
            data = pickle.load(f)

        # 强制都转成 float32，确保训练稳定
        self.features = data[0].astype(np.float32)   # (N, 11, T, V)
        self.adj_raw  = data[1].astype(np.float32)   # (N, V, V)
        self.mean_xy  = data[2].astype(np.float32)   # (N, 2)

        self.N = len(self.features)

        # -------------------
        # 构造 Graph 工具
        # -------------------
        self.graph = Graph(**graph_args)

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        feat = self.features[idx]           # (11, T, V)
        adj  = self.adj_raw[idx]            # (120, 120)
        mean = self.mean_xy[idx]            # (2,)

        # Graph normalized adjacency
        A = self.graph.get_adjacency(adj)       # (K, 120, 120)
        A = self.graph.normalize_adjacency(A)   # (K, 120, 120)

        # DataLoader 会自动把 numpy → torch
        return feat, A.astype(np.float32), mean
    