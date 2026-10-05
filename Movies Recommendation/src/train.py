from pathlib import Path

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
    n_factors=20,
    lr=0.01,
    reg=0.02
)


# ============================================================
# TRAIN MODEL
# ============================================================

print("Starting training...")

model.fit(
    user_idx,
    item_idx,
    r,
    epochs=25,
    validation_split=0.2,
    patience=3
)


print("Training finished!")


# ============================================================
# SAVE MODEL
# ============================================================

np.savez(
    MODEL_PATH,
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


# ============================================================
# SAVE TRAINING HISTORY
# ============================================================

np.savez(
    HISTORY_PATH,
    train_rmse=np.array(
        model.train_rmse_history
    ),
    val_rmse=np.array(
        model.val_rmse_history
    )
)


print("Saved model_weights.npz")
print("Saved training_history.npz")

print("Training pipeline completed!")