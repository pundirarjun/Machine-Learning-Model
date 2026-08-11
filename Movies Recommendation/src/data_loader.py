import pandas as pd
import numpy as np

def load_data(ratings_path=r"D:\Machine Learning Model\Movies Recommendation\data\ratings.csv", movies_path=r"D:\Machine Learning Model\Movies Recommendation\data\movies.csv"):
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