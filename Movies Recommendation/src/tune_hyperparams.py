"""Quick hyperparameter sweep for the matrix factorization model.

Each config uses the same seed so the train/validation split is
identical across runs and the numbers are directly comparable.

Results are written to ``tune_results.json`` after every config, so
an interrupted sweep resumes where it stopped instead of retraining
everything:

    python src/tune_hyperparams.py           # resume / run all
    python src/tune_hyperparams.py --reset   # start from scratch
"""

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from data_loader import load_data  # noqa: E402
from model import MatrixFactorization  # noqa: E402


PROJECT_ROOT = Path(__file__).resolve().parent.parent

RESULTS_PATH = PROJECT_ROOT / "tune_results.json"

SEED = 42

EPOCHS = 60

VALIDATION_SPLIT = 0.2

CONFIGS = [
    {
        "name": "k20 reg0.02 decay0.95 (current)",
        "n_factors": 20,
        "reg": 0.02,
        "lr": 0.01,
        "lr_decay": 0.95,
    },
    {
        "name": "k20 reg0.05 decay0.95",
        "n_factors": 20,
        "reg": 0.05,
        "lr": 0.01,
        "lr_decay": 0.95,
    },
    {
        "name": "k20 reg0.10 decay0.95",
        "n_factors": 20,
        "reg": 0.10,
        "lr": 0.01,
        "lr_decay": 0.95,
    },
    {
        "name": "k40 reg0.05 decay0.95",
        "n_factors": 40,
        "reg": 0.05,
        "lr": 0.01,
        "lr_decay": 0.95,
    },
    {
        "name": "k40 reg0.02 decay0.95",
        "n_factors": 40,
        "reg": 0.02,
        "lr": 0.01,
        "lr_decay": 0.95,
    },
    {
        "name": "k64 reg0.02 decay0.95",
        "n_factors": 64,
        "reg": 0.02,
        "lr": 0.01,
        "lr_decay": 0.95,
    },
    {
        "name": "k64 reg0.05 decay0.95",
        "n_factors": 64,
        "reg": 0.05,
        "lr": 0.01,
        "lr_decay": 0.95,
    },
    {
        "name": "k128 reg0.02 decay0.95",
        "n_factors": 128,
        "reg": 0.02,
        "lr": 0.01,
        "lr_decay": 0.95,
    },
    {
        "name": "k128 reg0.05 decay0.95",
        "n_factors": 128,
        "reg": 0.05,
        "lr": 0.01,
        "lr_decay": 0.95,
    },
    {
        "name": "k20 reg0.02 constant lr (no decay)",
        "n_factors": 20,
        "reg": 0.02,
        "lr": 0.01,
        "lr_decay": 1.0,
    },
]


def load_results():
    """Return {config name: record} of already-finished runs."""

    if not RESULTS_PATH.exists():
        return {}

    try:
        data = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        print(f"WARNING: could not read {RESULTS_PATH.name} ({exc}); starting fresh.")
        return {}

    return {rec["name"]: rec for rec in data.get("results", [])}


def save_results(results):
    """Persist every finished run so an interrupted sweep can resume."""

    tmp = RESULTS_PATH.with_suffix(".json.tmp")
    tmp.write_text(
        json.dumps({"results": list(results.values())}, indent=2),
        encoding="utf-8",
    )
    tmp.replace(RESULTS_PATH)


def main():

    if "--reset" in sys.argv and RESULTS_PATH.exists():
        RESULTS_PATH.unlink()

    results = load_results()

    if results:
        print(f"Resuming: {len(results)} config(s) already finished.")

    print("Loading data...")

    ratings, _movies, user2idx, movie2idx, _idx2movie = load_data()

    user_idx = ratings["user_idx"].values.astype(np.int32)
    item_idx = ratings["movie_idx"].values.astype(np.int32)
    r = ratings["rating"].values.astype(np.float32)

    for cfg in CONFIGS:

        if cfg["name"] in results:
            print(f"Skipping (already done): {cfg['name']}")
            continue

        print()
        print("=" * 60)
        print(cfg["name"])
        print("=" * 60)

        # Seed before init so weights start the same way
        # (per factor count), and again before fit so the
        # train/validation split is identical for every config.
        np.random.seed(SEED)

        model = MatrixFactorization(
            n_users=len(user2idx),
            n_items=len(movie2idx),
            n_factors=cfg["n_factors"],
            lr=cfg["lr"],
            reg=cfg["reg"],
        )

        np.random.seed(SEED)

        started = time.time()

        model.fit(
            user_idx,
            item_idx,
            r,
            epochs=EPOCHS,
            validation_split=VALIDATION_SPLIT,
            patience=EPOCHS,  # disable early stopping
            verbose=False,
            lr_decay=cfg["lr_decay"],
        )

        elapsed = time.time() - started

        best_val = float(np.min(model.val_rmse_history))
        final_train = float(model.train_rmse_history[-1])

        results[cfg["name"]] = {
            "name": cfg["name"],
            "n_factors": cfg["n_factors"],
            "reg": cfg["reg"],
            "lr": cfg["lr"],
            "lr_decay": cfg["lr_decay"],
            "best_val_rmse": best_val,
            "final_train_rmse": final_train,
            "seconds": elapsed,
            "best_epoch": int(np.argmin(model.val_rmse_history)) + 1,
        }

        save_results(results)

        print(
            f"best val RMSE {best_val:.4f} | "
            f"final train {final_train:.4f} | "
            f"{elapsed:.0f}s"
        )

    print()
    print("=" * 60)
    print("RANKING (lower validation RMSE is better)")
    print("=" * 60)

    ranked = sorted(results.values(), key=lambda rec: rec["best_val_rmse"])

    for rank, rec in enumerate(ranked, start=1):
        print(
            f"{rank}. {rec['best_val_rmse']:.4f}  "
            f"train {rec['final_train_rmse']:.4f}  "
            f"ep{rec['best_epoch']:>3d}  "
            f"{rec['seconds']:4.0f}s  {rec['name']}"
        )


if __name__ == "__main__":
    main()
