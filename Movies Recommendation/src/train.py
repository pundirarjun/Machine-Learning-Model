from pathlib import Path

import os

import numpy as np

from data_loader import load_data
from model import MatrixFactorization


# ============================================================
# CONFIG
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = PROJECT_ROOT / "model_weights.npz"

HISTORY_PATH = PROJECT_ROOT / "training_history.npz"

POPULARITY_PATH = PROJECT_ROOT / "movie_popularity.npy"

SEED = 42

# ------------------------------------------------------------
# Training schedule
#
# The previous run stopped at epoch 25 with val RMSE still
# decreasing, so the model was under-trained. We now train up
# to 60 epochs with an exponential learning-rate decay and a
# slightly longer early-stopping patience.
# ------------------------------------------------------------

N_FACTORS = 64

EPOCHS = 60

BASE_LR = 0.01

LR_DECAY = 0.95

MIN_LR = 1e-4

REG = 0.02

VALIDATION_SPLIT = 0.2

PATIENCE = 5


np.random.seed(SEED)


# ============================================================
# LOAD DATA
# ============================================================

print("Loading data...")

ratings, movies, user2idx, movie2idx, idx2movie = load_data()

print("Data loaded!")

print(
    "Number of ratings:",
    len(ratings)
)

print(
    "Number of users:",
    len(user2idx)
)

print(
    "Number of movies:",
    len(movie2idx)
)


# ============================================================
# MOVIE POPULARITY
# ============================================================

print("Calculating movie popularity...")

movie_rating_counts = (
    ratings
    .groupby("movie_idx")
    .size()
    .reindex(
        range(len(movie2idx)),
        fill_value=0
    )
    .values
    .astype(np.int32)
)

np.save(
    POPULARITY_PATH,
    movie_rating_counts
)

print("Saved movie_popularity.npy")


# ============================================================
# PREPARE TRAINING ARRAYS
# ============================================================

user_idx = (
    ratings["user_idx"]
    .values
    .astype(np.int32)
)

item_idx = (
    ratings["movie_idx"]
    .values
    .astype(np.int32)
)

r = (
    ratings["rating"]
    .values
    .astype(np.float32)
)


# ============================================================
# CREATE MODEL
# ============================================================

print("Creating model...")

model = MatrixFactorization(
    n_users=len(user2idx),
    n_items=len(movie2idx),
    n_factors=N_FACTORS,
    lr=BASE_LR,
    reg=REG
)


# ============================================================
# TRAIN MODEL
# ============================================================

print("Starting training...")

print(
    f"Schedule: {EPOCHS} epochs, "
    f"lr={BASE_LR} * {LR_DECAY}^epoch "
    f"(min {MIN_LR}), "
    f"reg={REG}, "
    f"k={N_FACTORS}, "
    f"split={VALIDATION_SPLIT}, "
    f"patience={PATIENCE}"
)

model.fit(
    user_idx,
    item_idx,
    r,
    epochs=EPOCHS,
    validation_split=VALIDATION_SPLIT,
    patience=PATIENCE,
    lr_decay=LR_DECAY,
    min_lr=MIN_LR,
    verbose=True
)


print("Training finished!")


# ============================================================
# SAVE MODEL
# ============================================================

# Write to a temporary file first so an interrupted run can
# never leave a half-written model_weights.npz behind.

tmp_model_path = MODEL_PATH.with_name(
    "model_weights.tmp.npz"
)

np.savez(
    tmp_model_path,
    P=model.P,
    Q=model.Q,
    b_u=model.b_u,
    b_i=model.b_i,
    global_mean=model.global_mean,
    movie_ids=np.array(
        [
            idx2movie[i]
            for i in range(len(idx2movie))
        ],
        dtype=np.int32
    )
)

os.replace(tmp_model_path, MODEL_PATH)


# ============================================================
# SAVE TRAINING HISTORY
# ============================================================

tmp_history_path = HISTORY_PATH.with_name(
    "training_history.tmp.npz"
)

np.savez(
    tmp_history_path,
    train_rmse=np.array(
        model.train_rmse_history
    ),
    val_rmse=np.array(
        model.val_rmse_history
    ),
    lr=np.array(
        model.lr_history
    )
)

os.replace(tmp_history_path, HISTORY_PATH)


print("Saved model_weights.npz")
print("Saved training_history.npz")

print("Training pipeline completed!")