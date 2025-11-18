# traj_dataset.py
import pickle
import numpy as np
from torch.utils.data import Dataset

from layers.graph import Graph


class TrajDataset(Dataset):
    """
    Clean Feeder:

    data_path 里是一个 pickle：
        [features, adjacency, mean_xy]

        features : (N, 11, T, V)
        adjacency: (N, V, V)
        mean_xy  : (N, 2)
    """

    def __init__(self, data_path, graph_args=None):
        super().__init__()

        self.data_path = data_path
        graph_args = graph_args or {"max_hop": 2, "num_node": 120}

        with open(data_path, "rb") as f:
            feats, adjs, means = pickle.load(f)

        # 统一转成 float32，避免后面出现 float64 / double
        self.features = feats.astype(np.float32)   # (N, 11, T, V)
        self.adj_raw  = adjs.astype(np.float32)    # (N, V, V)
        self.mean_xy  = means.astype(np.float32)   # (N, 2)

        self.N = self.features.shape[0]

        self.graph = Graph(**graph_args)

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        feat = self.features[idx]    # (11, T, V)
        adj  = self.adj_raw[idx]     # (V, V)
        mean = self.mean_xy[idx]     # (2,)

        # Graph adjacency: (K, V, V)
        A = self.graph.get_adjacency(adj)
        A = self.graph.normalize_adjacency(A)

        # 再保险，强制 float32
        A = A.astype(np.float32)
        feat = feat.astype(np.float32)
        mean = mean.astype(np.float32)

        return feat, A, mean
