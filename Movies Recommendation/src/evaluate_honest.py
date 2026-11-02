from pathlib import Path

import numpy as np

from data_loader import load_data


PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = PROJECT_ROOT / "model_weights.npz"

TEST_SPLIT_PATH = PROJECT_ROOT / "test_ratings.npz"

K = 10

MIN_RATINGS = 20

N_USERS = 1000

RELEVANT_RATING = 4.0

SEED = 42


print("Loading data...")

ratings, movies, user2idx, movie2idx, idx2movie = load_data()

weights = np.load(MODEL_PATH)

P = weights["P"]
Q = weights["Q"]

b_u = weights["b_u"]
b_i = weights["b_i"]

global_mean = float(weights["global_mean"])


test_data = np.load(TEST_SPLIT_PATH)

test_indices = test_data["test_indices"]

test_ratings = ratings.iloc[test_indices]

print(f"Test ratings: {len(test_ratings):,}")


user_pos = test_ratings.groupby("user_idx").indices

user_idx_arr = test_ratings["user_idx"].values.astype(np.int32)

movie_idx_arr = test_ratings["movie_idx"].values.astype(np.int32)

rating_arr = test_ratings["rating"].values.astype(np.float32)


train_user_idx = ratings.drop(ratings.index[test_indices])["user_idx"].values.astype(np.int32)

train_movie_idx = ratings.drop(ratings.index[test_indices])["movie_idx"].values.astype(np.int32)

train_seen = {}
for u, m in zip(train_user_idx, train_movie_idx):
    if u not in train_seen:
        train_seen[u] = set()
    train_seen[u].add(m)


eligible_users = np.array(
    [user for user, pos in user_pos.items() if len(pos) >= MIN_RATINGS],
    dtype=np.int32
)

np.random.seed(SEED)

if len(eligible_users) > N_USERS:
    test_users = np.random.choice(eligible_users, N_USERS, replace=False)
else:
    test_users = eligible_users

print(f"Evaluating {len(test_users)} users")


precision_scores = []
recall_scores = []
ndcg_scores = []


for user in test_users:

    pos = user_pos[user].copy()

    relevant = set(
        movie_idx_arr[pos][rating_arr[pos] >= RELEVANT_RATING].tolist()
    )

    if len(relevant) == 0:
        continue

    seen = train_seen.get(user, set())

    user_vector = P[user]

    scores = (
        global_mean
        + b_u[user]
        + b_i
        + Q.dot(user_vector)
    )

    if seen:
        scores[np.fromiter(seen, dtype=np.int64)] = -np.inf

    effective_k = min(K, len(scores))

    top_k = np.argpartition(scores, -effective_k)[-effective_k:]

    top_k = top_k[np.argsort(scores[top_k])[::-1]]

    recommended = top_k.tolist()

    hits = len(set(recommended) & relevant)

    precision = hits / effective_k
    precision_scores.append(precision)

    recall = hits / len(relevant)
    recall_scores.append(recall)

    dcg = 0.0
    for rank, movie_idx in enumerate(recommended, start=1):
        if movie_idx in relevant:
            dcg += 1 / np.log2(rank + 1)

    ideal_hits = min(len(relevant), effective_k)
    idcg = sum(1 / np.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    ndcg = dcg / idcg if idcg > 0 else 0
    ndcg_scores.append(ndcg)


print()
print("=" * 40)

if not precision_scores:
    print("No eligible users evaluated.")
else:
    print(f"Users evaluated: {len(precision_scores)}")
    print(f"Precision@{K}: {np.mean(precision_scores):.4f}")
    print(f"Recall@{K}: {np.mean(recall_scores):.4f}")
    print(f"NDCG@{K}: {np.mean(ndcg_scores):.4f}")

print("=" * 40)
print("HONEST metrics: evaluated on held-out test set never seen during training.")