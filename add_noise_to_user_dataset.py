import argparse
import pickle
import numpy as np
from pathlib import Path

def ensure_dir(path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--sigma_xy", type=float, default=1.0)
    parser.add_argument("--sigma_heading", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    np.random.seed(args.seed)
    ensure_dir(args.output)

    print(f"Loading clean dataset: {args.input}")
    feat, adj, mean_xy = pickle.load(open(args.input, "rb"))

    # noisy feat (copy)
    noisy = feat.copy()

    # add noise to x/y
    noise_x = np.random.normal(0, args.sigma_xy, size=noisy[:, 3].shape)
    noise_y = np.random.normal(0, args.sigma_xy, size=noisy[:, 4].shape)

    noisy[:, 3] += noise_x
    noisy[:, 4] += noise_y

    # add noise to heading (optional)
    if args.sigma_heading > 0:
        noise_h = np.random.normal(0, args.sigma_heading, size=noisy[:, 9].shape)
        noisy[:, 9] += noise_h

    # save
    with open(args.output, "wb") as f:
        pickle.dump([noisy, adj, mean_xy], f)

    print(f"[OK] saved noisy dataset → {args.output}")


if __name__ == "__main__":
    main()