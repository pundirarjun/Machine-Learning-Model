from pathlib import Path

import numpy as np

from data_loader import load_data


# ------------------------------------------------
# Configuration
# ------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = PROJECT_ROOT / "model_weights.npz"

K = 10

MIN_RATINGS = 20

N_USERS = 1000

RELEVANT_RATING = 4.0

SEED = 42


# ------------------------------------------------
# Load data
# ------------------------------------------------

print("Loading data...")

ratings, movies, user2idx, movie2idx, idx2movie = load_data()

weights = np.load(MODEL_PATH)

P = weights["P"]
Q = weights["Q"]

b_u = weights["b_u"]
b_i = weights["b_i"]

global_mean = float(
    weights["global_mean"]
)


# ------------------------------------------------
# Flatten ratings once
# ------------------------------------------------
#
# The old version filtered the full DataFrame once
# per user (1000 x 32M row scans). Grouping the
# positional indices once makes evaluation orders
# of magnitude faster.
#

user_pos = (
    ratings
    .groupby("user_idx")
    .indices
)

user_idx_arr = (
    ratings["user_idx"]
    .values
    .astype(np.int32)
)

movie_idx_arr = (
    ratings["movie_idx"]
    .values
    .astype(np.int32)
)

rating_arr = (
    ratings["rating"]
    .values
    .astype(np.float32)
)


# ------------------------------------------------
# Select users with enough ratings
# ------------------------------------------------

eligible_users = np.array(
    [
        user
        for user, pos in user_pos.items()
        if len(pos) >= MIN_RATINGS
    ],
    dtype=np.int32
)

np.random.seed(SEED)

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

    pos = user_pos[user].copy()

    # Shuffle ratings
    np.random.shuffle(pos)

    # Last 20% as test
    split = int(len(pos) * 0.8)

    train_pos = pos[:split]
    test_pos = pos[split:]

    if len(test_pos) == 0:
        continue

    # Relevant test movies
    relevant = set(
        movie_idx_arr[test_pos][
            rating_arr[test_pos] >= RELEVANT_RATING
        ].tolist()
    )

    if len(relevant) == 0:
        continue

    # Movies already seen during training
    seen = set(
        movie_idx_arr[train_pos].tolist()
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
    if seen:

        scores[
            np.fromiter(seen, dtype=np.int64)
        ] = -np.inf

    # Top K
    effective_k = min(K, len(scores))

    top_k = np.argpartition(
        scores,
        -effective_k
    )[-effective_k:]

    top_k = top_k[
        np.argsort(
            scores[top_k]
        )[::-1]
    ]

    recommended = top_k.tolist()

    # ------------------------------------------------
    # Precision@K
    # ------------------------------------------------

    hits = len(
        set(recommended) & relevant
    )

    precision = hits / effective_k

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
        effective_k
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

if not precision_scores:

    print("No eligible users evaluated.")

else:

    print(
        f"Users evaluated: "
        f"{len(precision_scores)}"
    )

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

print(
    "WARNING: this model was trained on ALL "
    "ratings, including the held-out ones used "
    "here. The numbers above are therefore "
    "OPTIMISTIC and must not be quoted as "
    "unbiased ranking accuracy. Retrain with a "
    "held-out split (or use cross-validation) "
    "for valid metrics."
)
