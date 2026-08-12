import numpy as np
import pandas as pd

from data_loader import load_data


# ------------------------------------------------
# Configuration
# ------------------------------------------------

K = 10

MIN_RATINGS = 20

N_USERS = 1000

RELEVANT_RATING = 4.0


# ------------------------------------------------
# Load data
# ------------------------------------------------

ratings, movies, user2idx, movie2idx, idx2movie = load_data()

weights = np.load(
    r"D:\Machine Learning Model\Movies Recommendation\model_weights.npz"
)

P = weights["P"]
Q = weights["Q"]

b_u = weights["b_u"]
b_i = weights["b_i"]

global_mean = float(
    weights["global_mean"]
)


# ------------------------------------------------
# Select users with enough ratings
# ------------------------------------------------

user_counts = (
    ratings
    .groupby("user_idx")
    .size()
)

eligible_users = user_counts[
    user_counts >= MIN_RATINGS
].index.values

np.random.seed(42)

if len(eligible_users) > N_USERS:

    test_users = np.random.choice(
        eligible_users,
        N_USERS,
        replace=False
    )

else:

    test_users = eligible_users


print(
    f"Evaluating {len(test_users)} users"
)


# ------------------------------------------------
# Metrics
# ------------------------------------------------

precision_scores = []

recall_scores = []

ndcg_scores = []


# ------------------------------------------------
# Evaluate users
# ------------------------------------------------

for user in test_users:

    user_ratings = ratings[
        ratings["user_idx"] == user
    ]

    # Shuffle ratings
    user_ratings = user_ratings.sample(
        frac=1,
        random_state=42
    )

    # Last 20% as test
    split = int(
        len(user_ratings) * 0.8
    )

    train_user = user_ratings.iloc[:split]

    test_user = user_ratings.iloc[split:]

    # Relevant test movies
    relevant = set(
        test_user[
            test_user["rating"] >= RELEVANT_RATING
        ]["movie_idx"]
    )

    if len(relevant) == 0:
        continue

    # Movies already seen
    seen = set(
        train_user["movie_idx"]
    )

    # User's learned factor
    user_vector = P[user]

    # Score all movies
    scores = (
        global_mean
        + b_u[user]
        + b_i
        + Q.dot(user_vector)
    )

    # Don't recommend already seen movies
    scores[
        list(seen)
    ] = -np.inf

    # Top K
    top_k = np.argpartition(
        scores,
        -K
    )[-K:]

    top_k = top_k[
        np.argsort(
            scores[top_k]
        )[::-1]
    ]

    recommended = list(top_k)

    # ------------------------------------------------
    # Precision@K
    # ------------------------------------------------

    hits = len(
        set(recommended) & relevant
    )

    precision = hits / K

    precision_scores.append(
        precision
    )

    # ------------------------------------------------
    # Recall@K
    # ------------------------------------------------

    recall = hits / len(relevant)

    recall_scores.append(
        recall
    )

    # ------------------------------------------------
    # NDCG@K
    # ------------------------------------------------

    dcg = 0.0

    for rank, movie_idx in enumerate(
        recommended,
        start=1
    ):

        if movie_idx in relevant:

            dcg += 1 / np.log2(
                rank + 1
            )

    # Ideal DCG
    ideal_hits = min(
        len(relevant),
        K
    )

    idcg = sum(
        1 / np.log2(rank + 1)
        for rank in range(
            1,
            ideal_hits + 1
        )
    )

    ndcg = (
        dcg / idcg
        if idcg > 0
        else 0
    )

    ndcg_scores.append(
        ndcg
    )


# ------------------------------------------------
# Final results
# ------------------------------------------------

print()
print("=" * 40)

print(
    f"Precision@{K}: "
    f"{np.mean(precision_scores):.4f}"
)

print(
    f"Recall@{K}: "
    f"{np.mean(recall_scores):.4f}"
)

print(
    f"NDCG@{K}: "
    f"{np.mean(ndcg_scores):.4f}"
)

print("=" * 40)