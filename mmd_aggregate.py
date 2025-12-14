#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
MMD-based robust aggregation of local models (log-median, spread-adaptive).

Core idea
---------
We measure how different each client's edge_importance vector is from a
reference edge vector using MMD, then convert these distances into
aggregation weights in a smooth, scale-invariant, and adaptive way:

  1) Compute MMD_i for each client i.
  2) Use a robust scale (median) to normalize:

       r_i = m_i / median(m)

  3) Work in log domain to get a scale-free notion of "how many times worse":

       log_r_i = log(r_i)

     - log_r_i ≈ 0  -> similar to median (typical client)
     - log_r_i  >>0 -> much worse than median (potential outlier)

  4) Only penalize clients that are worse than median:

       z_i = max(0, log_r_i)

     Clients with m_i <= median(m) get z_i = 0 (no penalty).

  5) Measure overall heterogeneity:

       spread = max(log_r_i) - min(log_r_i)

     If spread is small, all clients are similar and we should not overreact;
     if spread is large, there are strong outliers that should be heavily
     down-weighted.

  6) Adapt the effective penalty strength based on spread:

       alpha_eff = alpha * min(1, spread / s0)

     where s0 is a fixed scale threshold (in log space). When spread is small,
     alpha_eff << alpha and weights are close to uniform; when spread is large,
     alpha_eff ≈ alpha and strong outliers are exponentially suppressed.

  7) Map z_i -> weights via exponential decay:

       a_i = exp(-alpha_eff * z_i^power)
       w_i = a_i / sum_j a_j

     - alpha controls the maximum penalty strength.
     - power >= 1 controls how sharp the decay is (quadratic by default).

This is:
  - smooth (no hard cutoffs),
  - scale-invariant (depends on ratios m_i / median(m)),
  - adaptive: does not exaggerate tiny differences, but strongly penalizes
    clear outliers,
  - simple to tune: only (alpha, power) as hyperparameters.
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch


# -----------------------------
# I/O helpers
# -----------------------------
def ensure_dir(p):
    Path(p).mkdir(parents=True, exist_ok=True)


def load_model_state(p):
    """
    Load a model state_dict from a checkpoint.

    Expected formats:
      - {"xin_graph_seq2seq_model": state_dict}
      - state_dict directly
    """
    ckpt = torch.load(p, map_location="cpu")
    if isinstance(ckpt, dict) and "xin_graph_seq2seq_model" in ckpt:
        return ckpt["xin_graph_seq2seq_model"]
    elif isinstance(ckpt, dict):
        return ckpt
    else:
        raise ValueError(f"Unexpected model file format: {p}")


def save_model_state(state, out_path):
    """Save the aggregated state_dict in the same format as training."""
    ensure_dir(Path(out_path).parent)
    torch.save({"xin_graph_seq2seq_model": state}, out_path)


def load_edge_vector(npy_path):
    """
    Load edge_importance.npy and flatten to a 1D float64 vector.

    Supports shapes:
      [L, C, V, V], [L, V, V], [C, V, V], [V, V], or already flat [N].
    """
    arr = np.load(npy_path, allow_pickle=True)

    # Handle possible object-dtype wrapping
    if isinstance(arr, np.ndarray) and arr.dtype == object:
        arr = np.array(list(arr), dtype=object)
        if arr.dtype == object and len(arr) == 1:
            arr = arr[0]

    a = np.asarray(arr, dtype=np.float64)

    if a.ndim == 1:
        return a.copy()

    if a.ndim == 4:
        a = a.mean(axis=(0, 1))
    elif a.ndim == 3:
        a = a.mean(axis=0)
    elif a.ndim == 2:
        pass
    else:
        raise ValueError(f"Unexpected edge importance shape: {a.shape} in {npy_path}")

    return a.ravel()


# -----------------------------
# Blocked MMD (memory friendly)
# -----------------------------
def _rbf_block(a, b, gamma):
    # a:[p,1], b:[q,1] -> RBF kernel block [p,q]
    return np.exp(-gamma * (a - b.T) ** 2)


def compute_mmd_rbf(x, y, gamma=0.01, block=65536):
    """
    Estimate MMD^2 with an RBF kernel using blocking:

      MMD^2 = E[k(X,X')] + E[k(Y,Y')] - 2 E[k(X,Y)]

    x, y: 1D numpy arrays.
    """
    x = x.reshape(-1, 1).astype(np.float64, copy=False)
    y = y.reshape(-1, 1).astype(np.float64, copy=False)
    nx, ny = x.shape[0], y.shape[0]

    # E[k(X,X')]
    sx = 0.0
    cntx = 0
    for i in range(0, nx, block):
        xi = x[i:i + block]
        for j in range(0, nx, block):
            xj = x[j:j + block]
            k = _rbf_block(xi, xj, gamma)
            sx += k.sum()
            cntx += k.size
    exx = sx / cntx

    # E[k(Y,Y')]
    sy = 0.0
    cnty = 0
    for i in range(0, ny, block):
        yi = y[i:i + block]
        for j in range(0, ny, block):
            yj = y[j:j + block]
            k = _rbf_block(yi, yj, gamma)
            sy += k.sum()
            cnty += k.size
    eyy = sy / cnty

    # E[k(X,Y)]
    sxy = 0.0
    cntxy = 0
    for i in range(0, nx, block):
        xi = x[i:i + block]
        for j in range(0, ny, block):
            yj = y[j:j + block]
            k = _rbf_block(xi, yj, gamma)
            sxy += k.sum()
            cntxy += k.size
    exy = sxy / cntxy

    return float(exx + eyy - 2.0 * exy)


def auto_gamma_from_vectors(ref_vec, user_vecs, max_samples=2000, seed=0):
    """
    Simple median-heuristic gamma selection.

    We concatenate ref_vec and all user_vecs into one long 1D array,
    randomly subsample up to `max_samples` entries, compute pairwise
    squared distances, and set:

        gamma = 1 / median(dist^2)

    This makes gamma scale-invariant and data-dependent.
    """
    rng = np.random.default_rng(seed)

    # Concatenate all vectors
    all_vec = np.concatenate([ref_vec] + list(user_vecs), axis=0)
    all_vec = all_vec.reshape(-1)

    n = all_vec.shape[0]
    if n <= 1:
        return 1.0

    # Subsample if too long, to keep O(n^2) manageable
    if n > max_samples:
        idx = rng.choice(n, size=max_samples, replace=False)
        sub = all_vec[idx].reshape(-1, 1)
    else:
        sub = all_vec.reshape(-1, 1)

    # Pairwise squared distances
    d2 = (sub - sub.T) ** 2
    # Drop zeros on the diagonal
    d2 = d2[d2 > 0]

    if d2.size == 0:
        return 1.0

    med = float(np.median(d2))
    gamma = 1.0 / (med + 1e-12)
    return gamma


# -----------------------------
# Smooth MMD -> weights mapping (log-median, spread-adaptive)
# -----------------------------
def _maybe_subsample(v, subsample):
    if subsample is None or subsample <= 1:
        return v
    step = int(subsample)
    return v[::step]


def smooth_mmd_weights_log_median_adaptive(
    user_edge_vecs,
    ref_vec,
    gamma,
    eps=1e-12,
    block=65536,
    subsample=None,
    alpha=3.0,
    power=2.0,
):
    """
    Compute per-user MMD values and smooth aggregation weights.

    Steps:
      1) Compute MMD_i between each user edge and the reference.
      2) Let m_i = MMD_i and m_med = median(m_i).
      3) Normalize by the median:

           r_i = m_i / m_med

      4) Work in log domain:

           log_r_i = log(r_i)

      5) Measure global heterogeneity:

           spread = max(log_r_i) - min(log_r_i)

         If spread is small, all clients are similar and we should not
         overreact to tiny differences; if spread is large, strong outliers
         should be heavily down-weighted.

      6) Adapt the effective penalty strength based on spread:

           alpha_eff = alpha * min(1, spread / s0)

         where s0 is a fixed log-scale threshold (e.g., 0.25 ~ 28% ratio).
         When spread is small, alpha_eff << alpha and weights are close to
         uniform; when spread is large, alpha_eff ≈ alpha.

      7) Only penalize clients worse than median:

           z_i = max(0, log_r_i)

      8) Map z_i -> weights via exponential decay:

           a_i = exp(-alpha_eff * z_i^power)
           w_i = a_i / sum_j a_j
    """
    # 1) compute MMDs
    mmds = []
    for v in user_edge_vecs:
        vv = _maybe_subsample(v, subsample)
        rr = _maybe_subsample(ref_vec, subsample)
        m = compute_mmd_rbf(vv, rr, gamma=gamma, block=block)
        mmds.append(m)

    mmds = np.asarray(mmds, dtype=np.float64)
    N = mmds.shape[0]
    if N == 0:
        raise ValueError("No user edge vectors provided.")

    # Degenerate: all zero -> completely identical to reference
    if np.allclose(mmds, 0.0):
        weights = np.ones(N, dtype=np.float64) / N
        return [float(x) for x in mmds], weights

    # Median of MMDs
    m_med = float(np.median(mmds))
    if m_med <= 0.0:
        # Fallback: if median is non-positive, just use uniform weights
        w = np.ones(N, dtype=np.float64) / N
        return [float(x) for x in mmds], w

    alpha = float(max(alpha, 1e-8))
    power = float(max(power, 1.0))

    # 2) log ratios
    r = mmds / (m_med + eps)
    r = np.maximum(r, eps)
    log_r = np.log(r)

    # 3) spread in log space
    spread = float(log_r.max() - log_r.min())
    s0 = 0.25  # log-scale threshold (~ exp(0.25) ≈ 1.28)
    if s0 <= 0:
        alpha_eff = alpha
    else:
        scale = np.clip(spread / s0, 0.0, 1.0)
        alpha_eff = alpha * scale

    # 4) penalty for clients worse than median
    z = np.clip(log_r, 0.0, None)

    # 5) exponential decay
    a = np.exp(-alpha_eff * (z ** power))

    # Numerical safety
    if not np.isfinite(a).all():
        a = np.ones_like(a)

    s = a.sum()
    if s <= 0:
        weights = np.ones(N, dtype=np.float64) / N
    else:
        weights = a / s

    return [float(x) for x in mmds], weights


# -----------------------------
# Model / edge aggregation
# -----------------------------
def aggregate_states(states, weights):
    """
    Weighted average over a list of state_dicts.

    Only parameters that are shared across all models with the same
    shape are aggregated.
    """
    keys = set(states[0].keys())
    for st in states[1:]:
        keys &= set(st.keys())

    out = {}
    for k in keys:
        vs = [st[k] for st in states]
        shape0 = vs[0].shape
        if not all(v.shape == shape0 for v in vs):
            continue

        # Only aggregate floating-point tensors; for others, just copy the first one
        if all(torch.is_floating_point(v) for v in vs):
            # Stack parameters for this key: shape [N, ...]
            stacked = torch.stack(vs, dim=0)  # first dim = num_users
            w = torch.tensor(weights, dtype=stacked.dtype, device=stacked.device)

            # Build a broadcastable shape for w: [N, 1, 1, ..., 1]
            view_shape = [len(weights)] + [1] * (stacked.dim() - 1)
            w_view = w.view(*view_shape)

            # Weighted sum along user dimension
            weighted = (stacked * w_view).sum(dim=0)

            out[k] = weighted.to(vs[0].dtype)
        else:
            out[k] = vs[0]

    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--round", type=int, required=True, help="Round number")
    ap.add_argument("--ref_edge", type=str, required=True, help="Reference edge npy file")
    ap.add_argument("--user_models", type=str, nargs="+", required=True, help="User model checkpoints")
    ap.add_argument("--user_edges", type=str, nargs="+", required=True, help="User edge npy files")
    ap.add_argument("--out_model", type=str, required=True, help="Output aggregated model path")
    ap.add_argument("--out_edge", type=str, required=True, help="Output aggregated edge npy path")
    ap.add_argument("--out_metrics", type=str, required=True, help="Output metrics JSON path")
    ap.add_argument("--gamma", type=float, default=0.0, help="RBF kernel gamma for MMD")
    ap.add_argument("--block", type=int, default=65536, help="Block size for MMD computation")
    ap.add_argument("--eps", type=float, default=1e-12, help="Epsilon for numerical stability")
    ap.add_argument("--subsample", type=int, default=1, help="Subsampling step for edge vectors")
    ap.add_argument("--alpha", type=float, default=3.0,
                    help="Base penalty strength for MMD-based down-weighting.")
    ap.add_argument("--power", type=float, default=2.0,
                    help="Exponent in the decay exp(-alpha_eff * z^power).")

    args = ap.parse_args()

    user_vecs = [load_edge_vector(p) for p in args.user_edges]
    ref_vec = load_edge_vector(args.ref_edge)

    # Align vector lengths conservatively to the shortest among ref and users
    min_len = min([ref_vec.size] + [v.size for v in user_vecs])
    ref_vec = ref_vec[:min_len]
    user_vecs = [v[:min_len] for v in user_vecs]

    # Decide effective gamma: manual if >0, otherwise auto from data
    if args.gamma <= 0:
        gamma_eff = auto_gamma_from_vectors(ref_vec, user_vecs)
        print(f"[Mode] Auto gamma: {gamma_eff:.6f}")
    else:
        gamma_eff = float(args.gamma)
        print(f"[Mode] Manual gamma: {gamma_eff:.6f}")

    mmds, weights = smooth_mmd_weights_log_median_adaptive(
        user_edge_vecs=user_vecs,
        ref_vec=ref_vec,
        gamma=gamma_eff,
        eps=args.eps,
        block=args.block,
        subsample=None if args.subsample <= 1 else args.subsample,
        alpha=args.alpha,
        power=args.power,
    )

    states = [load_model_state(p) for p in args.user_models]
    agg_state = aggregate_states(states, weights)
    save_model_state(agg_state, args.out_model)

    # Weighted average of edges (using aligned 1D vectors)
    weighted_edge = np.zeros(min_len, dtype=np.float64)
    for v, w in zip(user_vecs, weights):
        weighted_edge += float(w) * v

    ensure_dir(Path(args.out_edge).parent)
    np.save(args.out_edge, weighted_edge)

    metrics = {
        "round": args.round,
        "ref_edge": args.ref_edge,
        "user_models": args.user_models,
        "user_edges": args.user_edges,
        "gamma": gamma_eff,
        "mmds": [float(x) for x in mmds],
        "weights": [float(w) for w in weights],
        "alpha": args.alpha,
        "power": args.power,
        "global_model": args.out_model,
        "global_edge": args.out_edge,
    }

    ensure_dir(Path(args.out_metrics).parent)
    with open(args.out_metrics, "w") as f:
        json.dump(metrics, f, indent=2)


if __name__ == "__main__":
    main()