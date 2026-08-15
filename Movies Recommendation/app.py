import streamlit as st
import numpy as np
import pandas as pd
import re
import requests

TMDB_API_KEY = st.secrets["TMDB_API_KEY"]

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Movie Recommender",
    layout="wide"
)

@st.cache_data(show_spinner=False)
def get_movie_poster(title):

    # Remove year from MovieLens title
    clean_title = re.sub(
        r"\s*\(\d{4}\)\s*$",
        "",
        title
    ).strip()

    url = "https://api.themoviedb.org/3/search/movie"

    params = {
        "api_key": TMDB_API_KEY,
        "query": clean_title,
        "language": "en-US",
        "include_adult": False
    }

    try:

        response = requests.get(
            url,
            params=params,
            timeout=5
        )

        if response.status_code != 200:
            return None

        data = response.json()

        results = data.get("results", [])

        if not results:
            return None

        poster_path = results[0].get(
            "poster_path"
        )

        if poster_path is None:
            return None

        return (
            "https://image.tmdb.org/t/p/w500"
            + poster_path
        )

    except Exception:
        return None

# ============================================================
# LOAD MODEL + DATA
# ============================================================

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

    # ----------------------------------------
    # Movie ID ↔ model index
    # ----------------------------------------

    movie2idx = {
        int(movie_id): i
        for i, movie_id in enumerate(movie_ids)
    }

    idx2movie = {
        i: int(movie_id)
        for i, movie_id in enumerate(movie_ids)
    }

    # ----------------------------------------
    # Keep only movies present in the model
    # ----------------------------------------

    model_movies = movies[
        movies["movieId"].isin(movie_ids)
    ].copy()

    # Put movies in EXACT same order as Q
    model_movies["model_idx"] = (
        model_movies["movieId"].map(movie2idx)
    )

    model_movies = (
        model_movies
        .sort_values("model_idx")
        .reset_index(drop=True)
    )

    # ----------------------------------------
    # Popularity
    # ----------------------------------------

    popularity = np.load(
        r"D:\Machine Learning Model\Movies Recommendation\movie_popularity.npy"
    )

    return (
        movies,
        model_movies,
        Q,
        b_i,
        global_mean,
        movie2idx,
        idx2movie,
        popularity
    )


(
    movies,
    model_movies,
    Q,
    b_i,
    global_mean,
    movie2idx,
    idx2movie,
    popularity
) = load()

movie_info = movies.set_index("movieId")

# ============================================================
# MODEL INFORMATION
# ============================================================

n_factors = Q.shape[1]

# Normalize movie embeddings
Q_norm = Q / (
    np.linalg.norm(Q, axis=1, keepdims=True)
    + 1e-8
)


# ============================================================
# PREPARE MOVIE INFORMATION
# ============================================================

# Normalize titles
def normalize_title(title):

    title = title.lower().strip()

    # Remove year
    title = re.sub(
        r"\s*\(\d{4}\)\s*$",
        "",
        title
    )

    # Convert:
    # "Hangover, The"
    # into:
    # "The Hangover"

    match = re.match(
        r"^(.*),\s*(the|a|an)$",
        title
    )

    if match:

        title = (
            f"{match.group(2)} "
            f"{match.group(1)}"
        )

    # Remove punctuation
    title = re.sub(
        r"[^a-z0-9\s]",
        "",
        title
    )

    # Remove duplicate spaces
    title = re.sub(
        r"\s+",
        " ",
        title
    ).strip()

    return title


movies["normalized_title"] = (
    movies["title"]
    .apply(normalize_title)
)


# ============================================================
# TITLE SEARCH
# ============================================================

def find_movie_matches(title_query):

    q = normalize_title(title_query)

    if not q:
        return None

    # Exact match
    exact = movies[
        movies["normalized_title"] == q
    ]

    if len(exact) > 0:
        return exact.iloc[0]

    # Partial match
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

def get_recommendation_reasons(
    recommended_idx,
    liked_idxs,
    selected_genres
):

    reasons = []

    # ==========================================
    # 1. Similar to movies the user entered
    # ==========================================

    if liked_idxs:

        similarities = (
            Q_norm[liked_idxs]
            @ Q_norm[recommended_idx]
        )

        best_positions = np.argsort(
            similarities
        )[::-1][:2]

        similar_titles = []

        for position in best_positions:

            liked_idx = liked_idxs[position]

            liked_movie_id = idx2movie[
                liked_idx
            ]

            liked_title = movie_info.loc[
                liked_movie_id,
                "title"
            ]

            similar_titles.append(
                liked_title
            )

        if similar_titles:

            reasons.append(
                "Similar to "
                + ", ".join(
                    f"**{title}**"
                    for title in similar_titles
                )
            )

    # ==========================================
    # 2. Genre preference
    # ==========================================

    if selected_genres:

        movie_id = idx2movie[
            recommended_idx
        ]

        movie_genres = set(
            str(
                movie_info.loc[
                    movie_id,
                    "genres"
                ]
            ).split("|")
        )

        matched_genres = (
            movie_genres
            & set(selected_genres)
        )

        if matched_genres:

            reasons.append(
                "Matches your "
                + ", ".join(
                    f"**{genre}**"
                    for genre in matched_genres
                )
                + " preference"
            )

    return reasons

# ============================================================
# GENRE LIST
# ============================================================

all_genres = sorted(
    set(
        genre
        for genre_string
        in movies["genres"].dropna()

        for genre
        in genre_string.split("|")

        if genre != "(no genres listed)"
    )
)


# ============================================================
# STREAMLIT UI
# ============================================================

st.title("🎬 Movie Recommender")

st.write(
    "Tell us what you like — "
    "we'll find something for you."
)


# ============================================================
# 1. GENRE
# ============================================================

st.subheader(
    "1. Pick genres you enjoy"
)

selected_genres = st.multiselect(
    "Genres",
    all_genres,
    default=["Drama", "Comedy"]
)


# ============================================================
# 2. MOOD
# ============================================================

st.subheader(
    "2. What are you in the mood for?"
)

mood = st.selectbox(
    "Mood",
    [
        "Something popular",
        "Hidden gem / underrated",
        "Recent releases",
        "Classic / older films"
    ]
)


# ============================================================
# 3. MOVIES USER LIKES
# ============================================================

st.subheader(
    "3. Movies you've recently enjoyed "
    "(type names, one per line)"
)

recent_input = st.text_area(
    "e.g.\n"
    "Inception\n"
    "The Dark Knight\n"
    "Parasite",
    height=120
)


# ============================================================
# RECOMMENDATION
# ============================================================

if st.button(
    "Get Recommendations",
    type="primary"
):

    liked_movie_ids = []
    unmatched = []

    # --------------------------------------------------------
    # Resolve movie titles
    # --------------------------------------------------------

    for line in recent_input.splitlines():

        if not line.strip():
            continue

        match = find_movie_matches(line)

        if match is not None:

            liked_movie_ids.append(
                int(match["movieId"])
            )

        else:

            unmatched.append(line)


    # --------------------------------------------------------
    # Show unmatched titles
    # --------------------------------------------------------

    if unmatched:

        st.warning(
            "Couldn't find: "
            + ", ".join(unmatched)
        )


    # --------------------------------------------------------
    # Validate input
    # --------------------------------------------------------

    if not liked_movie_ids:

        st.warning(
            "Type at least one movie you liked."
        )

        st.stop()


    # --------------------------------------------------------
    # Convert movie IDs → model indices
    # --------------------------------------------------------

    idxs = [
        movie2idx[movie_id]
        for movie_id in liked_movie_ids
        if movie_id in movie2idx
    ]


    if not idxs:

        st.warning(
            "None of the movies could be "
            "matched to the trained model."
        )

        st.stop()


    # Movies user already entered
    exclude = set(idxs)


    # ========================================================
    # 1. USER PROFILE FROM LIKED MOVIES
    # ========================================================

    p_u = Q_norm[idxs].mean(axis=0)

    # Normalize user vector
    p_u = p_u / (
        np.linalg.norm(p_u)
        + 1e-8
    )


    # ========================================================
    # 2. CONTENT SIMILARITY
    # ========================================================

    content_score = Q_norm @ p_u


    # ========================================================
    # 3. GENRE SCORE
    # ========================================================

    selected_genres_set = set(
        selected_genres
    )


    def genre_score(genres):

        if not selected_genres_set:
            return 0.0

        movie_genres = set(
            genres.split("|")
        )

        matches = (
            movie_genres
            & selected_genres_set
        )

        return (
            len(matches)
            / len(selected_genres_set)
        )


    genre_scores = (
        model_movies["genres"]
        .fillna("")
        .apply(genre_score)
        .values
    )


    # ========================================================
    # 4. POPULARITY SCORE
    # ========================================================

    log_popularity = np.log1p(
        popularity
    )

    pop_score = (
        log_popularity
        - log_popularity.min()
    ) / (
        log_popularity.max()
        - log_popularity.min()
        + 1e-8
    )


    # ========================================================
    # 5. FINAL SCORE
    # ========================================================

    if mood == "Something popular":

        final_scores = (
            0.65 * content_score
            + 0.20 * genre_scores
            + 0.15 * pop_score
        )


    elif mood == "Hidden gem / underrated":

        final_scores = (
            0.80 * content_score
            + 0.20 * genre_scores
            - 0.10 * pop_score
        )


    else:

        final_scores = (
            0.75 * content_score
            + 0.25 * genre_scores
        )


    # ========================================================
    # 6. MOOD FILTER
    # ========================================================

    allowed_idx = np.ones(
        len(final_scores),
        dtype=bool
    )


    # IMPORTANT:
    # Use model_movies, NOT movies.
    #
    # model_movies has exactly 84,432 rows
    # matching Q and final_scores.

    years = (
        model_movies["title"]
        .str.extract(
            r"\((\d{4})\)"
        )[0]
        .astype(float)
    )


    if mood == "Recent releases":

        allowed_idx &= (
            years
            .fillna(0)
            .values >= 2015
        )


    elif mood == "Classic / older films":

        allowed_idx &= (
            years
            .fillna(9999)
            .values < 1990
        )


    # ========================================================
    # 7. REMOVE ALREADY LIKED MOVIES
    # ========================================================

    for idx in exclude:

        allowed_idx[idx] = False


    # ========================================================
    # 8. RANK MOVIES
    # ========================================================

    candidate_idx = np.where(
        allowed_idx
    )[0]


    ranked = candidate_idx[
        np.argsort(
            final_scores[candidate_idx]
        )[::-1]
    ]


    top_idx = ranked[:10]


    # ========================================================
    # 9. DISPLAY
    # ========================================================

    if len(top_idx) == 0:

        st.warning(
            "No movies matched these filters. "
            "Try different genres or mood."
        )

    else:

        st.subheader(
            "🎯 Recommended for you"
        )

        for rank, idx in enumerate(
            top_idx,
            start=1
        ):

            movie_id = idx2movie[idx]

            row = model_movies.iloc[idx]

            title = row["title"]
            genres = row["genres"]

            # ==========================================
            # GET POSTER
            # ==========================================

            poster_url = get_movie_poster(title)

            # ==========================================
            # GET EXPLANATION
            # ==========================================

            reasons = get_recommendation_reasons(
                idx,
                idxs,
                selected_genres
            )

            # ==========================================
            # DISPLAY
            # ==========================================

            col1, col2 = st.columns(
                [1, 4]
            )

            with col1:

                if poster_url:

                    st.image(
                        poster_url,
                        width=140
                    )

                else:

                    st.markdown(
                        "🎬\n\nPoster unavailable"
                    )

            with col2:

                st.markdown(
                    f"### {rank}. ⭐ {title}"
                )

                st.write(
                    f"**Genres:** {genres}"
                )

                # Explanation
                for reason in reasons:

                    st.markdown(
                        f"🎯 {reason}"
                    )

                st.caption(
                    f"Recommendation score: "
                    f"{final_scores[idx]:.3f}"
                )

            st.divider()

        
    # ============================================================
# TMDB ATTRIBUTION
# ============================================================

st.divider()

st.caption(
    "This product uses the TMDB API but is not endorsed "
    "or certified by TMDB."
)