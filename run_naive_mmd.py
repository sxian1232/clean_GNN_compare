import argparse
import numpy as np


# -------------------------
# Core math
# -------------------------

def mmd2_rbf_singleton_fixed_sigma(x, y, sigma, eps=1e-12):
    """
    Singleton MMD^2 with RBF kernel and FIXED sigma:
      MMD^2 = 2 - 2 * exp(-||x-y||^2 / (2*sigma^2))
    """
    sigma = max(float(sigma), eps)
    dist2 = float(np.sum((x - y) ** 2))
    kxy = float(np.exp(-dist2 / (2.0 * sigma * sigma)))
    return 2.0 - 2.0 * kxy


def softmax_weights_from_mmd(mmds, gamma=1.0, eps=1e-12):
    """
    w_i = softmax(-gamma * mmd_i)
    """
    users = list(mmds.keys())
    vals = np.array([mmds[u] for u in users], dtype=np.float64)

    logits = -gamma * vals
    logits -= np.max(logits)  # stability
    exps = np.exp(logits)
    ws = exps / (np.sum(exps) + eps)
    return {u: float(w) for u, w in zip(users, ws)}


def power_sharpen(weights, p=1.0, eps=1e-12):
    """
    Optional: w' = normalize(w^p). If p>1, small weights shrink more.
    """
    if p is None or abs(p - 1.0) < 1e-12:
        return weights

    users = list(weights.keys())
    ws = np.array([weights[u] for u in users], dtype=np.float64)
    ws2 = np.power(ws, float(p))
    ws2 = ws2 / (ws2.sum() + eps)
    return {u: float(w) for u, w in zip(users, ws2)}


# -------------------------
# IO helpers
# -------------------------

def load_edge_vec(path: str) -> np.ndarray:
    """
    Load edge npy and flatten to a vector.
    user edges are (4,3,120,120), global_edge is (14400,) but we don't use it as ref here.
    """
    return np.load(path).astype(np.float64).reshape(-1)


def build_prev_global_by_mean(prev_round_dir: str, users=("user1","user2","user3","user4","user5")) -> np.ndarray:
    """
    Build 172800-dim 'prev_global' surrogate by averaging prev-round user edges.
    This avoids mismatch with compact global_edge.npy (14400).
    """
    prev_edges = []
    for u in users:
        p = f"{prev_round_dir}/{u}_edge.npy"
        prev_edges.append(load_edge_vec(p))
    return np.mean(np.stack(prev_edges, axis=0), axis=0)


# -------------------------
# Gamma (Auto) strategies
# -------------------------

def compute_sigma_from_ds(ds, eps=1e-12):
    """
    sigma used inside RBF MMD: we use median(d) as a fixed scale for the round.
    """
    d_vals = np.array(list(ds.values()), dtype=np.float64)
    return float(np.median(d_vals))


def compute_cv(ds, eps=1e-12):
    """
    Coefficient of variation: std(d)/mean(d) (dimensionless)
    """
    d_vals = np.array(list(ds.values()), dtype=np.float64)
    return float(np.std(d_vals) / (np.mean(d_vals) + eps))


def compute_gamma(ds, sigma, k=10.0, mode="cv", eps=1e-12):
    """
    AutoGamma (per round, no history):

    mode:
      - 'fixed'     : gamma = k
      - 'inv_sigma' : gamma = k / sigma
      - 'cv'        : gamma = k * cv
      - 'cv_inv'    : gamma = (k * cv) / sigma

    推荐：cv（最符合你想要的“差异小就平，差异大就狠”）
    """
    cv = compute_cv(ds, eps=eps)

    if mode == "fixed":
        gamma = float(k)
    elif mode == "inv_sigma":
        gamma = float(k) / (sigma + eps)
    elif mode == "cv":
        gamma = float(k) * cv
    elif mode == "cv_inv":
        gamma = (float(k) * cv) / (sigma + eps)
    else:
        raise ValueError("gamma_mode must be one of: fixed, inv_sigma, cv, cv_inv")

    return gamma, cv


# -------------------------
# Main runner
# -------------------------

def run_one_round(round_idx: int, root="trained/mmd",
                  users=("user1","user2","user3","user4","user5"),
                  k=10.0, gamma_mode="cv", p=1.0):

    curr_dir = f"{root}/r{round_idx}"
    prev_dir = f"{root}/r{round_idx-1}"

    # 1) prev global surrogate (same dim as user edges)
    g = build_prev_global_by_mean(prev_dir, users=users)

    # 2) current locals
    local = {u: load_edge_vec(f"{curr_dir}/{u}_edge.npy") for u in users}

    # sanity shape
    for u, x in local.items():
        assert x.shape == g.shape, f"Shape mismatch: {u} {x.shape} vs ref {g.shape}"

    # 3) distances
    ds = {u: float(np.linalg.norm(local[u] - g)) for u in users}

    # 4) fixed sigma for MMD kernel
    sigma = compute_sigma_from_ds(ds)

    # 5) MMDs
    mmds = {u: mmd2_rbf_singleton_fixed_sigma(local[u], g, sigma=sigma) for u in users}

    # 6) AutoGamma
    gamma, cv = compute_gamma(ds, sigma, k=k, mode=gamma_mode)

    # 7) weights
    weights = softmax_weights_from_mmd(mmds, gamma=gamma)

    # 8) optional power sharpening
    weights2 = power_sharpen(weights, p=p)

    # print
    print(f"\n=== Round{round_idx} naive MMD -> weights ===")
    print(f"(ref=mean(prev_round user edges), sigma=median(d)={sigma:.6f}, "
          f"cv={cv:.6f}, gamma_mode={gamma_mode}, k={k}, gamma={gamma:.6f}, p={p})\n")

    print(" user  |    d=||x-g||    |    mmd2    |   w_base   |  w_final")
    for u in users:
        print(f"{u:>5} | {ds[u]:>13.6f} | {mmds[u]:>9.6f} | {weights[u]:>9.6f} | {weights2[u]:>9.6f}")

    rank = sorted(users, key=lambda u: weights2[u], reverse=True)
    print("\nWeight ranking (high -> low):", " > ".join(rank))

    return {
        "round": round_idx,
        "sigma": sigma,
        "cv": cv,
        "gamma": gamma,
        "ds": ds,
        "mmds": mmds,
        "weights_base": weights,
        "weights_final": weights2,
        "ranking": rank,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--round", type=int, required=True, help="Current round index, e.g. 2/3/4/5")
    ap.add_argument("--root", type=str, default="trained/mmd", help="Root dir containing r{n}/ folders")
    ap.add_argument("--k", type=float, default=10.0, help="AutoGamma coefficient k")
    ap.add_argument("--gamma_mode", type=str, default="cv",
                    choices=["fixed", "inv_sigma", "cv", "cv_inv"],
                    help="AutoGamma mode (recommended: cv)")
    ap.add_argument("--p", type=float, default=1.0, help="Power sharpening exponent (1.0 disables)")
    args = ap.parse_args()

    run_one_round(
        round_idx=args.round,
        root=args.root,
        k=args.k,
        gamma_mode=args.gamma_mode,
        p=args.p,
    )


if __name__ == "__main__":
    main()