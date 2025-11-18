# generate_all_user_rounds.py
import argparse
import pickle
import json
import numpy as np
from pathlib import Path


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def add_noise(feat, sigma_xy, seed):
    """Return noisy version of feature data."""
    if sigma_xy == 0:
        return feat.copy()

    rng = np.random.default_rng(seed)

    noisy = feat.copy()

    # x/y noise
    noisy[:, 3] += rng.normal(0, sigma_xy, size=noisy[:, 3].shape)
    noisy[:, 4] += rng.normal(0, sigma_xy, size=noisy[:, 4].shape)

    return noisy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--clean_dir", type=str, default="data/processed/users_clean")
    parser.add_argument("--out_dir", type=str, default="data/processed/users_rounds")
    parser.add_argument("--rounds", type=int, required=True)
    parser.add_argument("--seed_base", type=int, default=1000,
                        help="Base seed, final seed = seed_base + round * 10 + user")
    args = parser.parse_args()

    # user noise settings
    user_noise = {
        1: 2.0,
        2: 1.0,
        3: 0.0,
        4: 0.0,
    5: 0.0,
    }

    global_metadata = {}  # save all users all rounds

    for r in range(1, args.rounds + 1):
        print(f"\n======== GENERATING ROUND {r} ========")
        round_dir = Path(args.out_dir) / f"round{r}"
        ensure_dir(round_dir)

        round_meta = {}

        for user in range(1, 5 + 1):
            clean_path = Path(args.clean_dir) / f"user{user}_clean_train.pkl"
            feat, adj, mean_xy = pickle.load(open(clean_path, "rb"))

            sigma = user_noise[user]

            # *** seed definition for reproducibility ***
            # Example: seed = 1000 + round * 10 + user
            seed = args.seed_base + r * 10 + user

            noisy_feat = add_noise(feat, sigma, seed)

            # file naming
            if sigma == 0:
                name = f"user{user}_clean.pkl"
            else:
                name = f"user{user}_u{sigma}_r{r}.pkl"

            out_path = round_dir / name

            pickle.dump([noisy_feat, adj, mean_xy], open(out_path, "wb"))
            print(f"Saved → {out_path}")

            # ----- save metadata -----
            round_meta[f"user{user}"] = {
                "round": r,
                "user": user,
                "sigma": sigma,
                "seed": seed,
                "clean_source": str(clean_path),
                "output": str(out_path),
            }

        # save metadata for this round
        with open(round_dir / "metadata.json", "w") as f:
            json.dump(round_meta, f, indent=2)

        global_metadata[f"round{r}"] = round_meta

    # save global metadata
    global_json_path = Path(args.out_dir) / "all_rounds_metadata.json"
    with open(global_json_path, "w") as f:
        json.dump(global_metadata, f, indent=2)

    print("\nAll rounds generated successfully!")
    print("Global metadata saved →", global_json_path)


if __name__ == "__main__":
    main()