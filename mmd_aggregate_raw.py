#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
from pathlib import Path

import numpy as np
import torch


# -----------------------------
# Utils
# -----------------------------
def ensure_parent(path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def load_model_state(p: str):
    ckpt = torch.load(p, map_location="cpu")
    if isinstance(ckpt, dict) and "xin_graph_seq2seq_model" in ckpt:
        return ckpt["xin_graph_seq2seq_model"]
    elif isinstance(ckpt, dict):
        return ckpt
    else:
        raise ValueError(f"Unexpected checkpoint format: {p}")


def save_model_state(state, out_path: str):
    ensure_parent(out_path)
    torch.save({"xin_graph_seq2seq_model": state}, out_path)


def load_edge_raw(path: str) -> np.ndarray:
    """Load raw edge importance (no pooling). Returns float64 np.ndarray with original shape."""
    arr = np.load(path, allow_pickle=True)
    if isinstance(arr, np.ndarray) and arr.dtype == object:
        # unwrap possible object container
        if len(arr) == 1:
            arr = arr[0]
        else:
            arr = np.array(list(arr), dtype=object)
            if len(arr) == 1:
                arr = arr[0]
    a = np.asarray(arr, dtype=np.float64)
    return a


def flatten_edge(a: np.ndarray) -> np.ndarray:
    return a.reshape(-1)


def coef_var(x: np.ndarray, eps: float = 1e-12) -> float:
    x = np.asarray(x, dtype=np.float64)
    mu = float(np.mean(x))
    sd = float(np.std(x))
    return float(sd / (mu + eps))


# -----------------------------
# Naive singleton MMD^2 (RBF)
# -----------------------------
def mmd2_rbf_singleton(x: np.ndarray, y: np.ndarray, sigma: float, eps: float = 1e-12) -> float:
    """
    Singleton estimate:
      MMD^2 = k(x,x) + k(y,y) - 2k(x,y)
            = 2 - 2 * exp(-||x-y||^2 / (2 sigma^2))
    """
    sigma = max(float(sigma), eps)
    dist2 = float(np.sum((x - y) ** 2))
    kxy = float(np.exp(-dist2 / (2.0 * sigma * sigma)))
    return 2.0 - 2.0 * kxy


def softmax_from_scores(scores: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """scores: larger means better. Return normalized weights."""
    s = scores - np.max(scores)
    exps = np.exp(s)
    return exps / (np.sum(exps) + eps)


# -----------------------------
# Model aggregation
# -----------------------------
def aggregate_states(states, weights):
    """
    Weighted average over a list of state_dicts.
    Only aggregate keys existing in all states with the same shape.
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
            stacked = torch.stack(vs, dim=0)  # [N, ...]
            w = torch.tensor(weights, dtype=stacked.dtype, device=stacked.device)
            view_shape = [len(weights)] + [1] * (stacked.dim() - 1)
            w_view = w.view(*view_shape)
            out[k] = (stacked * w_view).sum(dim=0).to(vs[0].dtype)
        else:
            out[k] = vs[0]

    return out


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--round", type=int, required=True)
    ap.add_argument("--ref_edge", type=str, required=True)
    ap.add_argument("--user_models", type=str, nargs="+", required=True)
    ap.add_argument("--user_edges", type=str, nargs="+", required=True)

    ap.add_argument("--out_model", type=str, required=True)
    ap.add_argument("--out_edge", type=str, required=True)
    ap.add_argument("--out_metrics", type=str, required=True)

    # gamma strategy (your current preferred one)
    ap.add_argument("--gamma_mode", type=str, default="cv",
                    choices=["fixed", "cv", "inv_sigma"],
                    help="fixed: gamma=k; cv: gamma=k*cv(d); inv_sigma: gamma=k*(1/sigma)")
    ap.add_argument("--k", type=float, default=10.0, help="scale factor for gamma")
    ap.add_argument("--p", type=float, default=1.0, help="optional power on mmd2 before softmax")
    ap.add_argument("--eps", type=float, default=1e-12)

    args = ap.parse_args()

    if len(args.user_models) != len(args.user_edges):
        raise ValueError("user_models and user_edges must have the same length.")

    users = [Path(p).stem.replace(".pt", "") for p in args.user_models]

    # 1) load raw edges (must be same shape)
    ref_raw = load_edge_raw(args.ref_edge)
    ref_vec = flatten_edge(ref_raw)

    user_raw = []
    user_vecs = []
    for p in args.user_edges:
        a = load_edge_raw(p)
        if a.shape != ref_raw.shape:
            raise ValueError(f"Edge shape mismatch:\n  user={p} shape={a.shape}\n  ref ={args.ref_edge} shape={ref_raw.shape}")
        user_raw.append(a)
        user_vecs.append(flatten_edge(a))

    # 2) compute distances d_i = ||x_i - ref||
    ds = np.array([np.linalg.norm(v - ref_vec) for v in user_vecs], dtype=np.float64)

    # 3) sigma = median(d)
    sigma = float(np.median(ds))
    if not np.isfinite(sigma) or sigma <= 0:
        sigma = float(np.mean(ds) + args.eps)

    # 4) cv on d
    cv = coef_var(ds, eps=args.eps)

    # 5) gamma
    if args.gamma_mode == "fixed":
        gamma = float(args.k)
    elif args.gamma_mode == "inv_sigma":
        gamma = float(args.k) * (1.0 / (sigma + args.eps))
    else:  # "cv"
        gamma = float(args.k) * float(cv)

    # 6) naive singleton MMD^2
    mmd2 = np.array([mmd2_rbf_singleton(v, ref_vec, sigma=sigma, eps=args.eps) for v in user_vecs], dtype=np.float64)

    # 7) weights via softmax(-gamma * mmd2^p)
    p = float(max(args.p, 1e-12))
    penal = mmd2 ** p
    scores = -gamma * penal  # lower mmd2 => higher score
    weights = softmax_from_scores(scores, eps=args.eps)

    # 8) aggregate model
    states = [load_model_state(p) for p in args.user_models]
    agg_state = aggregate_states(states, weights)
    save_model_state(agg_state, args.out_model)

    # 9) aggregate raw edge (same shape as ref)
    # compute in flattened space, then reshape back to raw shape for saving
    weighted_vec = np.zeros_like(ref_vec, dtype=np.float64)
    for v, w in zip(user_vecs, weights):
        weighted_vec += float(w) * v
    weighted_raw = weighted_vec.reshape(ref_raw.shape)

    ensure_parent(args.out_edge)
    np.save(args.out_edge, weighted_raw.astype(np.float32))
    # (存 float32 就够了，省空间；你要 float64 也可以删掉 astype)

    # 10) dump metrics
    metrics = {
        "round": args.round,
        "ref_edge": args.ref_edge,
        "user_models": args.user_models,
        "user_edges": args.user_edges,
        "edge_shape": list(ref_raw.shape),

        "sigma_median_d": sigma,
        "cv_d": cv,
        "gamma_mode": args.gamma_mode,
        "k": args.k,
        "gamma": gamma,
        "p": args.p,

        "d_norms": [float(x) for x in ds],
        "mmd2": [float(x) for x in mmd2],
        "weights": [float(x) for x in weights],

        "out_model": args.out_model,
        "out_edge": args.out_edge,
    }

    ensure_parent(args.out_metrics)
    with open(args.out_metrics, "w") as f:
        json.dump(metrics, f, indent=2)

    # 11) nice print for screenshot
    print(f"\n=== Round{args.round} naive MMD -> weights (RAW ONLY) ===")
    print(f"(ref={args.ref_edge}, edge_shape={ref_raw.shape}, sigma=median(d)={sigma:.6f}, "
          f"cv={cv:.6f}, gamma_mode={args.gamma_mode}, k={args.k}, gamma={gamma:.6f}, p={args.p})\n")
    print(f"{'user':>6} | {'d=||x-ref||':>14} | {'mmd2':>10} | {'w':>10}")
    for i, u in enumerate(users):
        print(f"{u:>6} | {ds[i]:14.6f} | {mmd2[i]:10.6f} | {weights[i]:10.6f}")

    order = np.argsort(-weights)
    ranking = " > ".join([users[i] for i in order])
    print("\nWeight ranking (high -> low):", ranking)


if __name__ == "__main__":
    main()