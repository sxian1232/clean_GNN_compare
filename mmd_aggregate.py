import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch


# ======================================================
# I/O utilities
# ======================================================

def ensure_dir(p):
    Path(p).mkdir(parents=True, exist_ok=True)


def load_model_state(path):
    """Load model state_dict from checkpoint."""
    ckpt = torch.load(path, map_location="cpu")
    if isinstance(ckpt, dict) and "xin_graph_seq2seq_model" in ckpt:
        return ckpt["xin_graph_seq2seq_model"]
    elif isinstance(ckpt, dict):
        return ckpt
    else:
        raise ValueError(f"Unexpected checkpoint format: {path}")


def save_model_state(state, out_path):
    ensure_dir(Path(out_path).parent)
    torch.save({"xin_graph_seq2seq_model": state}, out_path)


def load_edge_vector(npy_path):
    """
    Load edge importance from .npy and return a 1D flattened vector.
    Supports shapes:
        [L, C, V, V], [L, V, V], [C, V, V], [V, V], [N]
    """
    arr = np.load(npy_path, allow_pickle=True)

    # Handle object arrays (e.g., list stored in numpy)
    if isinstance(arr, np.ndarray) and arr.dtype == object:
        arr = np.array(list(arr))

    a = np.asarray(arr, dtype=np.float64)

    if a.ndim == 1:
        return a.copy()
    elif a.ndim == 4:
        a = a.mean(axis=(0, 1))
    elif a.ndim == 3:
        a = a.mean(axis=0)
    elif a.ndim == 2:
        pass
    else:
        raise ValueError(f"Unexpected shape for edge importance: {a.shape}")

    return a.ravel()


# ======================================================
# MMD computation
# ======================================================

def _rbf_block(a, b, gamma):
    """RBF kernel block for vectors a,b."""
    return np.exp(-gamma * (a - b.T) ** 2)


def compute_mmd_rbf(x, y, gamma, block=65536):
    """
    Block-wise unbiased MMD^2 estimate:
        MMD^2 = E[k(X,X')] + E[k(Y,Y')] - 2 E[k(X,Y)]
    """
    x = x.reshape(-1, 1).astype(np.float64)
    y = y.reshape(-1, 1).astype(np.float64)
    nx, ny = x.shape[0], y.shape[0]

    # E[k(X,X')]
    sx, cntx = 0.0, 0
    for i in range(0, nx, block):
        xi = x[i:i+block]
        for j in range(0, nx, block):
            xj = x[j:j+block]
            k = _rbf_block(xi, xj, gamma)
            sx += k.sum()
            cntx += k.size
    exx = sx / cntx

    # E[k(Y,Y')]
    sy, cnty = 0.0, 0
    for i in range(0, ny, block):
        yi = y[i:i+block]
        for j in range(0, ny, block):
            yj = y[j:j+block]
            k = _rbf_block(yi, yj, gamma)
            sy += k.sum()
            cnty += k.size
    eyy = sy / cnty

    # E[k(X,Y)]
    sxy, cntxy = 0.0, 0
    for i in range(0, nx, block):
        xi = x[i:i+block]
        for j in range(0, ny, block):
            yj = y[j:j+block]
            k = _rbf_block(xi, yj, gamma)
            sxy += k.sum()
            cntxy += k.size
    exy = sxy / cntxy

    return float(exx + eyy - 2 * exy)


def mmd_weights(user_vecs, ref_vec, gamma, eps=1e-12, block=65536, subsample=1):
    """
    Compute MMD weights for each user:
        w_i ∝ 1 / (MMD_i + eps)
    """
    def maybe_subsample(v):
        if subsample <= 1:
            return v
        return v[::subsample]

    ref = maybe_subsample(ref_vec)
    mmds = []
    for v in user_vecs:
        vv = maybe_subsample(v)
        m = compute_mmd_rbf(vv, ref, gamma=gamma, block=block)
        mmds.append(m)

    inv = np.array([1.0 / (m + eps) for m in mmds])
    w = inv / inv.sum()

    return mmds, w


def auto_gamma(vecs):
    """Median heuristic: gamma = 1 / median(||x - y||^2)"""
    all_vec = np.concatenate(vecs)
    N = min(2000, len(all_vec))
    idx = np.random.choice(len(all_vec), N, replace=False)
    sub = all_vec[idx].reshape(-1, 1)
    d = (sub - sub.T) ** 2
    med = np.median(d[d > 0])
    gamma = 1.0 / (med + 1e-12)
    return gamma


# ======================================================
# Aggregation
# ======================================================

def aggregate_states(states, weights):
    """Weighted average of model state_dicts."""
    keys = set(states[0].keys())
    for st in states[1:]:
        keys &= set(st.keys())

    out = {}

    for k in keys:
        vals = [s[k] for s in states]
        shape0 = vals[0].shape

        if not all(v.shape == shape0 for v in vals):
            continue

        if all(torch.is_floating_point(v) for v in vals):
            acc = None
            for w, v in zip(weights, vals):
                vv = v.float() * float(w)
                acc = vv if acc is None else acc + vv
            out[k] = acc.to(vals[0].dtype)
        else:
            out[k] = vals[0]

    return out


def aggregate_edges(edge_paths, weights, save_path=None):
    """Weighted average of flattened edge vectors."""
    vecs = [load_edge_vector(p) for p in edge_paths]
    L = min(v.size for v in vecs)
    vecs = [v[:L] for v in vecs]

    acc = np.zeros(L, dtype=np.float64)
    for v, w in zip(vecs, weights):
        acc += w * v

    if save_path:
        ensure_dir(Path(save_path).parent)
        np.save(save_path, acc)

    return acc


# ======================================================
# Main
# ======================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--round", type=int, required=True)
    ap.add_argument("--ref_edge", type=str, required=True)
    ap.add_argument("--user_models", nargs="+", required=True)
    ap.add_argument("--user_edges", nargs="+", required=True)

    ap.add_argument("--gamma", type=float, default=0.0,
                    help="0 = auto gamma (median heuristic); >0 = manual gamma")

    ap.add_argument("--subsample", type=int, default=1)
    ap.add_argument("--block", type=int, default=65536)

    ap.add_argument("--out_dir", type=str, default=None)

    args = ap.parse_args()

    assert len(args.user_models) == len(args.user_edges), "Mismatch: models vs edges"

    # Output directory
    out_dir = args.out_dir or f"trained/mmd/r{args.round}"
    ensure_dir(out_dir)

    # ----------------------------------------------------
    # Load reference edge (global from previous round)
    # ----------------------------------------------------
    ref_vec = load_edge_vector(args.ref_edge)

    # Load user edges
    user_vecs = [load_edge_vector(p) for p in args.user_edges]

    # Align vector lengths
    L = min([ref_vec.size] + [v.size for v in user_vecs])
    ref_vec = ref_vec[:L]
    user_vecs = [v[:L] for v in user_vecs]

    # ----------------------------------------------------
    # Determine gamma
    # ----------------------------------------------------
    if args.gamma == 0:
        # Auto gamma
        gamma = auto_gamma([ref_vec] + user_vecs)
        print(f"[Gamma] Auto gamma = {gamma:.4e}")
    else:
        gamma = args.gamma
        print(f"[Gamma] Manual gamma = {gamma:.4e}")

    # ----------------------------------------------------
    # Compute MMD and weights
    # ----------------------------------------------------
    mmds, weights = mmd_weights(
        user_vecs, ref_vec, gamma=gamma,
        eps=1e-12, block=args.block, subsample=args.subsample
    )

    print("[MMD]", mmds)
    print("[Weights]", weights)

    # ----------------------------------------------------
    # Aggregate model parameters
    # ----------------------------------------------------
    print("[Model] Loading user models...")
    states = [load_model_state(p) for p in args.user_models]

    print("[Model] Aggregating...")
    agg_state = aggregate_states(states, weights)

    out_model = os.path.join(out_dir, f"global.pt")
    save_model_state(agg_state, out_model)
    print("[Model] Saved →", out_model)

    # ----------------------------------------------------
    # Aggregate edge importance
    # ----------------------------------------------------
    out_edge = os.path.join(out_dir, f"global_edge.npy")
    aggregate_edges(args.user_edges, weights, save_path=out_edge)
    print("[Edge] Saved →", out_edge)

    # ----------------------------------------------------
    # Save metrics
    # ----------------------------------------------------
    metrics = {
        "round": args.round,
        "ref_edge": args.ref_edge,
        "user_models": args.user_models,
        "user_edges": args.user_edges,
        "gamma": gamma,
        "mmds": [float(m) for m in mmds],
        "weights": [float(w) for w in weights],
        "global_model": out_model,
        "global_edge": out_edge,
    }

    out_metrics = os.path.join(out_dir, f"global_metrics.json")
    with open(out_metrics, "w") as f:
        json.dump(metrics, f, indent=2)

    print("[Metrics] Saved →", out_metrics)
    print("\n===== MMD Aggregation Done =====")


if __name__ == "__main__":
    main()