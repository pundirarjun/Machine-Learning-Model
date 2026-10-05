from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"


def load_data(
    ratings_path=None,
    movies_path=None
):
    """
    Load MovieLens ratings and movies, then map
    raw IDs to contiguous model indices.
    """

    if ratings_path is None:
        ratings_path = DATA_DIR / "ratings.csv"

    if movies_path is None:
        movies_path = DATA_DIR / "movies.csv"

    ratings = pd.read_csv(ratings_path)
    movies = pd.read_csv(movies_path)

    user_ids = ratings["userId"].unique()
    movie_ids = ratings["movieId"].unique()

    user2idx = {u: i for i, u in enumerate(user_ids)}
    movie2idx = {m: i for i, m in enumerate(movie_ids)}
    idx2movie = {i: m for m, i in movie2idx.items()}

    ratings["user_idx"] = ratings["userId"].map(user2idx)
    ratings["movie_idx"] = ratings["movieId"].map(movie2idx)

    return ratings, movies, user2idx, movie2idx, idx2movie
