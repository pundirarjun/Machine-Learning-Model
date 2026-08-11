import numpy as np
from data_loader import load_data
from model import MatrixFactorization


print("Loading data...")

ratings, movies, user2idx, movie2idx, idx2movie = load_data()

print("Data loaded!")

print("Number of ratings:", len(ratings))
print("Number of users:", len(user2idx))
print("Number of movies:", len(movie2idx))


user_idx = ratings["user_idx"].values.astype(np.int32)
item_idx = ratings["movie_idx"].values.astype(np.int32)
r = ratings["rating"].values.astype(np.float32)


print("Creating model...")

model = MatrixFactorization(
    n_users=len(user2idx),
    n_items=len(movie2idx),
    n_factors=20,
    lr=0.01,
    reg=0.02
)


print("Starting training...")

model.fit(
    user_idx,
    item_idx,
    r,
    epochs=5
)


print("Training finished!")


np.savez(
    "model_weights.npz",
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

print("Saved model_weights.npz")