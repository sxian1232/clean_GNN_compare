#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
MMD-based robust aggregation of local models.

- Computes MMD between each user's edge_importance vector and a reference edge vector.
- Uses a piecewise, robust mapping from MMD values to aggregation weights:
    * If all clients are very similar (ratio <= tau_flat): use uniform weights.
    * If some strong outliers exist (ratio > tau_out): heavily down-weight outliers
      and nearly uniform among inliers.
    * Otherwise: interpolate between inverse-MMD weights and uniform weights.

This script does NOT implement FedAvg; you already have a separate script for that.
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

    We expect either:
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


# -----------------------------
# Robust MMD -> weights mapping
# -----------------------------
def _maybe_subsample(v, subsample):
    if subsample is None or subsample <= 1:
        return v
    step = int(subsample)
    return v[::step]


def robust_mmd_weights(
    user_edge_vecs,
    ref_vec,
    gamma,
    eps=1e-12,
    block=65536,
    subsample=None,
    tau_flat=1.2,
    tau_out=2.5,
    outlier_mass=0.03,
    lam=0.5,
):
    """
    Compute per-user MMD values and robust aggregation weights.

    Steps:
      1) Compute MMD_i between each user edge and the reference.
      2) Convert MMDs into a scale-free ratio: r_i = m_i / min(m).
      3) Piecewise mapping:

         - Case A: max(r_i) <= tau_flat
             -> users are very similar; use uniform weights.

         - Case B: max(r_i) > tau_out
             -> strong outliers exist.
                * mark inliers: r_i <= tau_out
                * assign almost all mass to inliers, and a small
                  total mass `outlier_mass` to outliers.

         - Case C: tau_flat < max(r_i) <= tau_out
             -> intermediate spread.
                * start from inverse-MMD weights
                * blend with uniform using parameter `lam`:
                    w = lam * w_inv_mmd + (1 - lam) * w_uniform

    Returns:
      mmds: list of float
      weights: numpy array of shape [num_users], sum to 1
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

    # numerical safety: if all MMDs are zero, just use uniform
    if np.allclose(mmds, 0.0):
        weights = np.ones(N, dtype=np.float64) / N
        return [float(x) for x in mmds], weights

    # 2) ratio relative to the best client
    m_min = float(mmds.min())
    ratios = mmds / (m_min + 1e-12)
    max_r = float(ratios.max())

    # clamp hyper-parameters to reasonable ranges
    tau_flat = max(float(tau_flat), 1.0)
    tau_out = max(float(tau_out), tau_flat + 1e-6)
    outlier_mass = float(np.clip(outlier_mass, 0.0, 0.5))
    lam = float(np.clip(lam, 0.0, 1.0))

    # -------- Case A: all clients very similar --------
    if max_r <= tau_flat:
        weights = np.ones(N, dtype=np.float64) / N

    # -------- Case B: strong outliers exist --------
    elif max_r > tau_out:
        inliers_mask = ratios <= tau_out
        num_in = int(inliers_mask.sum())
        num_out = N - num_in

        if num_in == 0 or num_out == 0:
            # Degenerate case: fall back to uniform
            weights = np.ones(N, dtype=np.float64) / N
        else:
            w_out = outlier_mass / num_out
            w_in = (1.0 - outlier_mass) / num_in

            weights = np.zeros(N, dtype=np.float64)
            weights[inliers_mask] = w_in
            weights[~inliers_mask] = w_out

    # -------- Case C: moderate spread --------
    else:
        inv = 1.0 / (mmds + eps)
        inv /= inv.sum()

        uniform = np.ones(N, dtype=np.float64) / N
        weights = lam * inv + (1.0 - lam) * uniform
        weights /= weights.sum()  # just in case of numerical drift

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

        if all(torch.is_floating_point(v) for v in vs):
            acc = None
            for v, w in zip(vs, weights):
                vv = v.float() * float(w)
                acc = vv if acc is None else (acc + vv)
            out[k] = acc.to(vs[0].dtype)
        else:
            # For non-float parameters (e.g., integer buffers) just take the first.
            out[k] = vs[0]
    return out


def weighted_edge_average(edge_paths, weights, save_path=None):
    """
    Weighted average of multiple edge_importance.npy vectors using
    the same aggregation weights.
    """
    vecs = [load_edge_vector(p) for p in edge_paths]
    L = min(v.size for v in vecs)
    vecs = [v[:L] for v in vecs]

    acc = np.zeros(L, dtype=np.float64)
    for v, w in zip(vecs, weights):
        acc += float(w) * v

    if save_path:
        ensure_dir(Path(save_path).parent)
        np.save(save_path, acc)
    return acc


# -----------------------------
# Gamma selection
# -----------------------------
def auto_gamma_from_vectors(ref_vec, user_vecs, max_samples=2000, seed=0):
    """
    Heuristic auto-gamma selection using the median of pairwise squared distances.

    We gather all entries from ref_vec and user_vecs, subsample up to `max_samples`,
    and set:
        gamma = 1 / median(dist^2)
    """
    rng = np.random.default_rng(seed)
    all_vec = np.concatenate([ref_vec] + list(user_vecs))
    all_vec = all_vec.reshape(-1)

    n = all_vec.shape[0]
    if n <= 1:
        return 1.0

    # subsample if very long
    if n > max_samples:
        idx = rng.choice(n, size=max_samples, replace=False)
        sub = all_vec[idx].reshape(-1, 1)
    else:
        sub = all_vec.reshape(-1, 1)

    d2 = (sub - sub.T) ** 2
    d2 = d2[d2 > 0]  # exclude zeros on diagonal
    if d2.size == 0:
        return 1.0

    med = float(np.median(d2))
    gamma = 1.0 / (med + 1e-12)
    return gamma


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--round", type=int, required=True,
                    help="Federated round index (for logging only).")
    ap.add_argument("--ref_edge", type=str, required=True,
                    help="Reference edge_importance.npy path "
                         "(typically previous global or initial clean).")
    ap.add_argument("--user_models", nargs="+", required=True,
                    help="List of local user model .pt paths.")
    ap.add_argument("--user_edges", nargs="+", required=True,
                    help="List of user edge_importance.npy paths "
                         "(same order as user_models).")

    # MMD / kernel hyper-parameters
    ap.add_argument("--gamma", type=float, default=0.0,
                    help="RBF kernel gamma. "
                         "0 means auto (median heuristic); >0 uses given value.")
    ap.add_argument("--block", type=int, default=65536,
                    help="Block size for MMD computation.")
    ap.add_argument("--eps", type=float, default=1e-12,
                    help="Small constant for numeric stability.")
    ap.add_argument("--subsample", type=int, default=1,
                    help="Subsample step for vectors (1 = no subsampling).")

    # Robust weighting hyper-parameters
    ap.add_argument("--tau_flat", type=float, default=1.2,
                    help="If max ratio <= tau_flat, use uniform weights.")
    ap.add_argument("--tau_out", type=float, default=2.5,
                    help="If max ratio > tau_out, treat large-ratio clients as outliers.")
    ap.add_argument("--outlier_mass", type=float, default=0.03,
                    help="Total weight mass assigned to all outliers.")
    ap.add_argument("--lam", type=float, default=0.5,
                    help="Blend factor between inverse-MMD and uniform "
                         "in the intermediate case (0~1).")

    # Output
    ap.add_argument("--out_dir", type=str, default=None,
                    help="Output directory for global model / metrics. "
                         "Default: trained/mmd/r{round}")

    args = ap.parse_args()

    if len(args.user_models) != len(args.user_edges):
        raise ValueError("user_models and user_edges must have the same length.")

    # Output directory
    out_dir = args.out_dir or f"trained/mmd/r{args.round}"
    ensure_dir(out_dir)

    # ---- 1) Load reference and user edge vectors ----
    print("[Info] Loading reference edge:", args.ref_edge)
    ref_vec = load_edge_vector(args.ref_edge)
    user_vecs = [load_edge_vector(p) for p in args.user_edges]

    # Align length (conservative: use the shortest)
    min_len = min([ref_vec.size] + [v.size for v in user_vecs])
    ref_vec = ref_vec[:min_len]
    user_vecs = [v[:min_len] for v in user_vecs]

    # ---- 2) Select gamma ----
    if args.gamma <= 0:
        gamma = auto_gamma_from_vectors(ref_vec, user_vecs)
        print(f"[Mode] Auto gamma: {gamma:.6f}")
    else:
        gamma = float(args.gamma)
        print(f"[Mode] Manual gamma: {gamma:.6f}")

    # ---- 3) Compute MMDs and robust weights ----
    print("[Info] Computing MMDs and robust weights ...")
    mmds, weights = robust_mmd_weights(
        user_edge_vecs=user_vecs,
        ref_vec=ref_vec,
        gamma=gamma,
        eps=args.eps,
        block=args.block,
        subsample=None if args.subsample <= 1 else args.subsample,
        tau_flat=args.tau_flat,
        tau_out=args.tau_out,
        outlier_mass=args.outlier_mass,
        lam=args.lam,
    )

    print("[Info] MMDs:", [float(x) for x in mmds])
    print("[Info] Weights (sum=1):", [float(w) for w in weights])

    # ---- 4) Load user models and aggregate ----
    print("[Info] Loading user models ...")
    states = [load_model_state(p) for p in args.user_models]
    agg_state = aggregate_states(states, weights)

    # ---- 5) Save aggregated global model ----
    out_model = os.path.join(out_dir, "global.pt")
    save_model_state(agg_state, out_model)
    print("[Info] Saved global model ->", out_model)

    # ---- 6) Weighted edge importance ----
    out_edge = os.path.join(out_dir, "global_edge.npy")
    _ = weighted_edge_average(args.user_edges, weights, save_path=out_edge)
    print("[Info] Saved global edge ->", out_edge)

    # ---- 7) Save metrics ----
    metrics = {
        "round": args.round,
        "ref_edge": args.ref_edge,
        "user_models": args.user_models,
        "user_edges": args.user_edges,
        "gamma": gamma,
        "mmds": [float(x) for x in mmds],
        "weights": [float(w) for w in weights],
        "tau_flat": args.tau_flat,
        "tau_out": args.tau_out,
        "outlier_mass": args.outlier_mass,
        "lam": args.lam,
        "global_model": out_model,
        "global_edge": out_edge,
    }
    out_metrics = os.path.join(out_dir, "global_metrics.json")
    with open(out_metrics, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print("[Info] Saved metrics ->", out_metrics)


if __name__ == "__main__":
    main()