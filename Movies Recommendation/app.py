import streamlit as st
import numpy as np
import pandas as pd
import re

st.set_page_config(page_title="Movie Recommender", layout="wide")

@st.cache_data
def load():

    movies = pd.read_csv(
        r"D:\Machine Learning Model\Movies Recommendation\data\movies.csv"
    )

    with np.load(
        r"D:\Machine Learning Model\Movies Recommendation\model_weights.npz"
    ) as weights:

        Q = weights["Q"]
        b_i = weights["b_i"]
        global_mean = float(weights["global_mean"])
        movie_ids = weights["movie_ids"]

    movie2idx = {
        int(movie_id): i
        for i, movie_id in enumerate(movie_ids)
    }

    idx2movie = {
        i: int(movie_id)
        for i, movie_id in enumerate(movie_ids)
    }

    # Movies that actually exist in the trained model
    model_movies = movies[
        movies["movieId"].isin(movie_ids)
    ].copy()

    # Put them in exactly the same order as Q
    model_movies["model_idx"] = (
        model_movies["movieId"].map(movie2idx)
    )

    model_movies = model_movies.sort_values(
        "model_idx"
    )

    return (
        movies,
        model_movies,
        Q,
        b_i,
        global_mean,
        movie2idx,
        idx2movie
    )

movies, model_movies,  Q, b_i, global_mean, movie2idx, idx2movie = load()
n_factors = Q.shape[1]
movie_info = movies.set_index("movieId")

model_movie_ids = np.array(
    [idx2movie[i] for i in range(len(Q))]
)

model_movies = movies[
    movies["movieId"].isin(model_movie_ids)
].copy()

model_movies["model_idx"] = model_movies["movieId"].map(movie2idx)

model_movies = model_movies.sort_values("model_idx")


# normalize Q rows for cosine similarity later
Q_norm = Q / (np.linalg.norm(Q, axis=1, keepdims=True) + 1e-8)

# extract genre list
all_genres = sorted(set(g for gs in movies["genres"].dropna() for g in gs.split("|") if g != "(no genres listed)"))

st.title("🎬 Movie Recommender")
st.write("Tell us what you like — we'll find something for you.")

# ---------- Step 1: Genre preferences ----------
st.subheader("1. Pick genres you enjoy")
selected_genres = st.multiselect("Genres", all_genres, default=["Drama", "Comedy"])

# ---------- Step 2: Mood / type ----------
st.subheader("2. What are you in the mood for?")
mood = st.selectbox(
    "Mood",
    ["Something popular", "Hidden gem / underrated", "Recent releases", "Classic / older films"]
)

# ---------- Step 3: Movies they've recently enjoyed (free text) ----------
st.subheader("3. Movies you've recently enjoyed (type names, one per line)")
recent_input = st.text_area(
    "e.g.\nInception\nThe Dark Knight\nParasite",
    height=100
)

# Using a normalized title search.

def normalize_title(title):

    title = title.lower().strip()

    # Remove year
    title = re.sub(r"\s*\(\d{4}\)\s*$", "", title)

    # Move trailing article to front
    match = re.match(
        r"^(.*),\s*(the|a|an)$",
        title
    )

    if match:
        title = f"{match.group(2)} {match.group(1)}"

    # Remove punctuation
    title = re.sub(r"[^a-z0-9\s]", "", title)

    # Normalize spaces
    title = re.sub(r"\s+", " ", title).strip()

    return title


movies["normalized_title"] = movies["title"].apply(
    normalize_title
)

def find_movie_matches(title_query):

    q = normalize_title(title_query)

    if not q:
        return None

    exact = movies[
        movies["normalized_title"] == q
    ]

    if len(exact) > 0:
        return exact.iloc[0]

    matches = movies[
        movies["normalized_title"].str.contains(
            q,
            na=False,
            regex=False
        )
    ]

    if len(matches) > 0:
        return matches.iloc[0]

    return None

# ---------- Step 4: Optional explicit ratings (kept as fallback) ----------

# with st.expander("Optional: rate a few sample movies instead"):
#     sample_movies = movies.sample(9, random_state=1)
#     user_ratings = {}
#     cols = st.columns(3)
#     for i, (_, row) in enumerate(sample_movies.iterrows()):
#         with cols[i % 3]:
#             r = st.slider(row["title"], 0, 5, 0, key=f"sample_{row['movieId']}")
#             if r > 0:
#                 user_ratings[row["movieId"]] = r

# ---------- Recommendation logic ----------
if st.button("Get Recommendations"):

    liked_movie_ids = []
    unmatched = []

    # Resolve typed movie names
    for line in recent_input.splitlines():

        if not line.strip():
            continue

        match = find_movie_matches(line)

        if match is not None:
            liked_movie_ids.append(match["movieId"])
        else:
            unmatched.append(line)

    if unmatched:
        st.warning(
            f"Couldn't find: {', '.join(unmatched)}"
        )

    if not liked_movie_ids:
        st.warning("Type at least one movie you liked.")
        st.stop()

    # Convert movie IDs to model indices
    idxs = [
        movie2idx[m]
        for m in liked_movie_ids
        if m in movie2idx
    ]

    if not idxs:
        st.warning("None of the movies could be matched to the trained model.")
        st.stop()

    exclude = set(idxs)

    # ------------------------------------------------
    # 1. Movie embedding profile
    # ------------------------------------------------

    p_u = Q_norm[idxs].mean(axis=0)

    # Normalize user profile
    p_u = p_u / (np.linalg.norm(p_u) + 1e-8)

    # Cosine similarity
    content_score = Q_norm @ p_u

    # ------------------------------------------------
    # 2. Genre score
    # ------------------------------------------------

    selected_genres_set = set(selected_genres)

    def genre_score(genres):

        if not selected_genres:
            return 0.0

        movie_genres = set(
            genres.split("|")
        )

        matches = movie_genres & set(selected_genres)

        return len(matches) / len(selected_genres)


    genre_scores = (
        model_movies["genres"]
        .fillna("")
        .apply(genre_score)
        .values
    )

    # ------------------------------------------------
    # 3. Combine scores
    # ------------------------------------------------

    final_scores = (
        0.75 * content_score + 0.25 * genre_scores
    )

    # ------------------------------------------------
    # 4. Apply mood
    # ------------------------------------------------

    allowed_idx = np.ones(
        len(final_scores),
        dtype=bool
    )

    if mood == "Recent releases":

        years = (
            movies["title"]
            .str.extract(r"\((\d{4})\)")[0]
            .astype(float)
        )

        allowed_idx &= (
            years.fillna(0).values >= 2015
        )

    elif mood == "Classic / older films":

        years = (
            movies["title"]
            .str.extract(r"\((\d{4})\)")[0]
            .astype(float)
        )

        allowed_idx &= (
            years.fillna(9999).values < 1990
        )

    # ------------------------------------------------
    # 5. Remove movies user already entered
    # ------------------------------------------------

    for idx in exclude:
        allowed_idx[idx] = False

    # ------------------------------------------------
    # 6. Get recommendations
    # ------------------------------------------------

    candidate_idx = np.where(allowed_idx)[0]

    ranked = candidate_idx[
        np.argsort(
            final_scores[candidate_idx]
        )[::-1]
    ]

    top_idx = ranked[:10]

    # ------------------------------------------------
    # 7. Display
    # ------------------------------------------------

    st.subheader("Recommended for you")

    for idx in top_idx:

        movie_id = idx2movie[idx]

        row = movies[
            movies["movieId"] == movie_id
        ].iloc[0]

        st.write(
            f"⭐ **{row['title']}** — "
            f"*{row['genres']}*"
        )