from pathlib import Path

import numpy as np

from data_loader import load_data
from model import MatrixFactorization


PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = PROJECT_ROOT / "model_weights.npz"

HISTORY_PATH = PROJECT_ROOT / "training_history.npz"

POPULARITY_PATH = PROJECT_ROOT / "movie_popularity.npy"

TEST_SPLIT_PATH = PROJECT_ROOT / "test_ratings.npz"

SEED = 42

N_FACTORS = 64

EPOCHS = 60

BASE_LR = 0.01

LR_DECAY = 0.95

MIN_LR = 1e-4

REG = 0.02

VALIDATION_SPLIT = 0.1

TEST_SPLIT = 0.1

PATIENCE = 5


np.random.seed(SEED)


print("Loading data...")

ratings, movies, user2idx, movie2idx, idx2movie = load_data()

print("Data loaded!")

print("Number of ratings:", len(ratings))

print("Number of users:", len(user2idx))

print("Number of movies:", len(movie2idx))


n_ratings = len(ratings)

indices = np.arange(n_ratings)

np.random.shuffle(indices)

test_size = int(n_ratings * TEST_SPLIT)

val_size = int(n_ratings * VALIDATION_SPLIT)

train_size = n_ratings - test_size - val_size

test_indices = indices[:test_size]

val_indices = indices[test_size:test_size + val_size]

train_indices = indices[test_size + val_size:]

print(f"Train: {len(train_indices):,}, Val: {len(val_indices):,}, Test: {len(test_indices):,}")


np.savez(TEST_SPLIT_PATH, test_indices=test_indices)


train_ratings = ratings.iloc[train_indices].reset_index(drop=True)

val_ratings = ratings.iloc[val_indices].reset_index(drop=True)

test_ratings = ratings.iloc[test_indices].reset_index(drop=True)


print("Calculating movie popularity...")

movie_rating_counts = (
    train_ratings
    .groupby("movie_idx")
    .size()
    .reindex(
        range(len(movie2idx)),
        fill_value=0
    )
    .values
    .astype(np.int32)
)

np.save(POPULARITY_PATH, movie_rating_counts)

print("Saved movie_popularity.npy")


user_idx = train_ratings["user_idx"].values.astype(np.int32)

item_idx = train_ratings["movie_idx"].values.astype(np.int32)

r = train_ratings["rating"].values.astype(np.float32)


val_user_idx = val_ratings["user_idx"].values.astype(np.int32)

val_item_idx = val_ratings["movie_idx"].values.astype(np.int32)

val_r = val_ratings["rating"].values.astype(np.float32)


print("Creating model...")

model = MatrixFactorization(
    n_users=len(user2idx),
    n_items=len(movie2idx),
    n_factors=N_FACTORS,
    lr=BASE_LR,
    reg=REG
)


print("Starting training...")

print(
    f"Schedule: {EPOCHS} epochs, "
    f"lr={BASE_LR} * {LR_DECAY}^epoch "
    f"(min {MIN_LR}), "
    f"reg={REG}, "
    f"k={N_FACTORS}, "
    f"val_split={VALIDATION_SPLIT}, "
    f"test_split={TEST_SPLIT}, "
    f"patience={PATIENCE}"
)


val_order = np.arange(len(val_r), dtype=np.int32)


model.fit(
    user_idx,
    item_idx,
    r,
    epochs=EPOCHS,
    validation_split=0.0,
    patience=PATIENCE,
    lr_decay=LR_DECAY,
    min_lr=MIN_LR,
    verbose=True,
    val_data=(val_user_idx, val_item_idx, val_r, val_order)
)


print("Training finished!")


tmp_model_path = MODEL_PATH.with_name("model_weights.tmp.npz")

np.savez(
    tmp_model_path,
    P=model.P,
    Q=model.Q,
    b_u=model.b_u,
    b_i=model.b_i,
    global_mean=model.global_mean,
    movie_ids=np.array(
        [idx2movie[i] for i in range(len(idx2movie))],
        dtype=np.int32
    )
)

import os
os.replace(tmp_model_path, MODEL_PATH)


tmp_history_path = HISTORY_PATH.with_name("training_history.tmp.npz")

np.savez(
    tmp_history_path,
    train_rmse=np.array(model.train_rmse_history),
    val_rmse=np.array(model.val_rmse_history),
    lr=np.array(model.lr_history)
)

os.replace(tmp_history_path, HISTORY_PATH)


print("Saved model_weights.npz")
print("Saved training_history.npz")
print("Training pipeline completed!")