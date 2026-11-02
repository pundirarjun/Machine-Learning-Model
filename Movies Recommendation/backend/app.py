from flask import Flask, jsonify, request
from flask_cors import CORS

import numpy as np
import pandas as pd
import os
import traceback
from pathlib import Path


# ============================================================
# CONFIG
# ============================================================

MAX_LIKED_MOVIES = 50

MOODS = (
    "Something popular",
    "Hidden gem / underrated",
    "Recent releases",
    "Classic / older films",
)

RECENT_YEAR_THRESHOLD = 2015

CLASSIC_YEAR_THRESHOLD = 1990

FLASK_DEBUG = os.getenv(
    "FLASK_DEBUG", "0"
) == "1"

FLASK_HOST = os.getenv(
    "FLASK_HOST", "127.0.0.1"
)

FLASK_PORT = int(
    os.getenv("FLASK_PORT", "5000")
)


# ============================================================
# FLASK APP
# ============================================================

app = Flask(__name__)


# ============================================================
# CORS
# ============================================================

CORS(
    app,
    resources={
        r"/*": {
            "origins": [
                "http://localhost:5173",
                "http://127.0.0.1:5173"
            ]
        }
    },
    methods=[
        "GET",
        "POST",
        "OPTIONS"
    ],
    allow_headers=[
        "Content-Type"
    ]
)


# ============================================================
# LOAD RECOMMENDATION MODEL
# ============================================================

print("Loading recommendation model...")

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from recommender import (
    movies,
    Q,
    Q_norm,
    movie2idx,
    idx2movie,
    movie_info,
    model_movies,
    popularity,
    normalize_title,
    get_movie_media,
    item_item_sim,
)

from item_item_cf import hybrid_scores

print("Recommendation model loaded!")
print(f"Movies in dataset: {len(movies):,}")
print(f"Movies in trained model: {len(Q):,}")


# ============================================================
# PREPARE MOVIE DATA
# ============================================================

# Work on a copy so we don't unexpectedly modify the
# DataFrame loaded by recommender.py.

movies = movies.copy()


# ------------------------------------------------------------
# Normalize movie titles
#
# normalize_title is imported from recommender.py so there is
# only one implementation of the title-cleaning rules.
# ------------------------------------------------------------

movies["normalized_title"] = (
    movies["title"]
    .apply(normalize_title)
)


# ============================================================
# MODEL MOVIE DATA
# ============================================================
#
# model_movies is imported from recommender.py, already
# filtered to the movies that exist in the trained model and
# sorted so that row order matches the Q matrix.
#

if len(model_movies) != len(Q):

    raise RuntimeError(
        "model_movies and Q are not aligned: "
        f"{len(model_movies)} != {len(Q)}"
    )


# ============================================================
# POPULARITY
# ============================================================

popularity = np.asarray(
    popularity,
    dtype=np.float32
)


if len(popularity) != len(Q):

    raise RuntimeError(
        "Popularity length does not match Q: "
        f"{len(popularity)} != {len(Q)}"
    )


# Normalize popularity
popularity_score = (
    popularity
    /
    (
        popularity.max()
        + 1e-8
    )
)


# Precomputed once: used by the "Hidden gem" mood
# filter and by the "Popular with viewers" reason.
POPULARITY_THRESHOLD = np.percentile(
    popularity,
    70
)


# ============================================================
# GENRE MASKS (PRECOMPUTED)
# ============================================================
#
# Previously every request looped over ~84k rows to score
# genres. One boolean mask per genre makes it O(#genres).
#

genre_masks = {}


for _, row in model_movies.iterrows():

    idx = int(
        row["model_idx"]
    )

    for genre in str(
        row["genres"]
    ).split("|"):

        if genre not in genre_masks:

            genre_masks[genre] = np.zeros(
                len(Q),
                dtype=bool
            )

        genre_masks[genre][idx] = True


# ============================================================
# YEAR EXTRACTION
# ============================================================

model_movies["year"] = (
    model_movies["title"]
    .str.extract(
        r"\((\d{4})\)"
    )[0]
    .astype(float)
)


# Create year array in EXACT Q order

year_by_idx = np.full(
    len(Q),
    np.nan,
    dtype=np.float32
)


for _, row in model_movies.iterrows():

    idx = int(
        row["model_idx"]
    )

    year_by_idx[idx] = (
        row["year"]
        if not pd.isna(row["year"])
        else np.nan
    )


# ============================================================
# TITLE MATCHING
# ============================================================

def find_movie_match(title_query):

    """
    Resolve a user's movie title to a movie row.

    Matching order:

    1. Exact normalized title
    2. Exact title contained in candidate
    3. Candidate title contained in query
    4. Prefer the most popular candidate
    """

    query = normalize_title(
        title_query
    )

    if not query:

        return None


    # --------------------------------------------------------
    # Exact normalized match
    # --------------------------------------------------------

    exact = movies[
        movies["normalized_title"] == query
    ]


    if len(exact) > 0:

        # Prefer movies that exist in trained model
        model_exact = exact[
            exact["movieId"].isin(
                movie2idx.keys()
            )
        ]

        if len(model_exact) > 0:
            exact = model_exact

        # If duplicates remain, use popularity
        if len(exact) > 1:

            candidates = []

            for _, row in exact.iterrows():

                movie_id = int(
                    row["movieId"]
                )

                if movie_id in movie2idx:

                    idx = movie2idx[
                        movie_id
                    ]

                    candidates.append(
                        (
                            popularity[idx],
                            row
                        )
                    )

            if candidates:

                candidates.sort(
                    key=lambda x: x[0],
                    reverse=True
                )

                return candidates[0][1]

        return exact.iloc[0]


    # --------------------------------------------------------
    # Partial matching
    # --------------------------------------------------------

    contains = movies[
        movies["normalized_title"]
        .str.contains(
            query,
            na=False,
            regex=False
        )
    ].copy()


    if len(contains) == 0:

        return None


    # Only movies represented by trained model
    contains = contains[
        contains["movieId"].isin(
            movie2idx.keys()
        )
    ]


    if len(contains) == 0:

        return None


    # --------------------------------------------------------
    # Rank partial matches
    # --------------------------------------------------------

    candidates = []


    for _, row in contains.iterrows():

        movie_id = int(
            row["movieId"]
        )

        idx = movie2idx[
            movie_id
        ]

        normalized = row[
            "normalized_title"
        ]


        # Match quality
        if normalized == query:

            match_score = 100

        elif normalized.startswith(query):

            match_score = 80

        elif query in normalized:

            match_score = 60

        else:

            match_score = 20


        # Popularity gives a small preference to well-known
        # versions when multiple movies share a name.
        pop = float(
            popularity[idx]
        )


        final_match_score = (
            match_score
            + min(
                pop / 1000.0,
                10
            )
        )


        candidates.append(
            (
                final_match_score,
                row
            )
        )


    if not candidates:

        return None


    candidates.sort(
        key=lambda x: x[0],
        reverse=True
    )


    return candidates[0][1]


# ============================================================
# GENRE SCORE
# ============================================================

def calculate_genre_scores(
    selected_genres
):

    scores = np.zeros(
        len(Q),
        dtype=np.float32
    )


    if not selected_genres:

        return scores


    selected = set(
        selected_genres
    )


    # Vectorised using the masks built at startup.
    #
    # A movie scores (matched genres / requested genres),
    # which is exactly what the previous per-row loop did.
    #

    counts = np.zeros(
        len(Q),
        dtype=np.float32
    )


    for genre in selected:

        mask = genre_masks.get(genre)

        if mask is None:

            continue

        counts[mask] += 1.0


    scores = (
        counts
        /
        float(len(selected))
    )


    return scores


# ============================================================
# MOOD FILTER
# ============================================================

def apply_mood_filter(
    mood,
    allowed_idx
):

    allowed = allowed_idx.copy()


    # --------------------------------------------------------
    # Recent
    # --------------------------------------------------------

    if mood == "Recent releases":

        allowed &= (
            year_by_idx >= RECENT_YEAR_THRESHOLD
        )


    # --------------------------------------------------------
    # Classic
    # --------------------------------------------------------

    elif mood == "Classic / older films":

        allowed &= (
            year_by_idx < CLASSIC_YEAR_THRESHOLD
        )


    # --------------------------------------------------------
    # Hidden gems
    # --------------------------------------------------------

    elif mood == "Hidden gem / underrated":

        # Lower popularity movies
        # Top 30% popularity are excluded.

        allowed &= (
            popularity < POPULARITY_THRESHOLD
        )


    # --------------------------------------------------------
    # Something popular
    # --------------------------------------------------------

    elif mood == "Something popular":

        # No hard filter.
        # Popularity is already part of score.
        pass


    return allowed


# ============================================================
# HOME
# ============================================================

@app.route(
    "/",
    methods=["GET"]
)
def home():

    return jsonify({
        "message":
            "Movie Recommendation API is running!",
        "movies":
            int(len(Q))
    })


# ============================================================
# HEALTH CHECK
# ============================================================

@app.route(
    "/health",
    methods=["GET"]
)
def health():

    return jsonify({

        "status":
            "ok",

        "model_movies":
            int(len(Q)),

        "dataset_movies":
            int(len(movies)),

        "model_factors":
            int(Q.shape[1])

    })


# ============================================================
# RECOMMEND
# ============================================================

@app.route(
    "/recommend",
    methods=[
        "POST",
        "OPTIONS"
    ]
)
def recommend():

    # --------------------------------------------------------
    # OPTIONS
    # --------------------------------------------------------

    if request.method == "OPTIONS":

        return jsonify({
            "status": "ok"
        })


    try:

        # ----------------------------------------------------
        # Read JSON
        # ----------------------------------------------------

        data = request.get_json(
            silent=True
        )


        if not data:

            return jsonify({
                "error":
                    "Request body is required."
            }), 400


        liked_movies = data.get(
            "movies",
            []
        )


        selected_genres = data.get(
            "genres",
            []
        )


        mood = data.get(
            "mood",
            "Something popular"
        )


        # ----------------------------------------------------
        # Validate movies
        # ----------------------------------------------------

        if not isinstance(
            liked_movies,
            list
        ):

            return jsonify({
                "error":
                    "movies must be a list."
            }), 400


        if len(liked_movies) == 0:

            return jsonify({
                "error":
                    "Please provide at least one movie."
            }), 400


        if len(liked_movies) > MAX_LIKED_MOVIES:

            return jsonify({
                "error":
                    f"Please provide at most "
                    f"{MAX_LIKED_MOVIES} movies."
            }), 400


        if not isinstance(
            selected_genres,
            list
        ):

            return jsonify({
                "error":
                    "genres must be a list."
            }), 400


        if mood not in MOODS:

            return jsonify({
                "error":
                    "Unknown mood."
            }), 400


        # ----------------------------------------------------
        # Resolve titles
        # ----------------------------------------------------

        liked_idxs = []

        matched_titles = []

        unmatched_titles = []


        for title in liked_movies:

            match = find_movie_match(
                title
            )


            if match is None:

                unmatched_titles.append(
                    title
                )

                continue


            movie_id = int(
                match["movieId"]
            )


            if movie_id not in movie2idx:

                unmatched_titles.append(
                    title
                )

                continue


            idx = movie2idx[
                movie_id
            ]


            # Avoid duplicate movies
            if idx not in liked_idxs:

                liked_idxs.append(
                    idx
                )

                matched_titles.append(
                    match["title"]
                )


        # ----------------------------------------------------
        # Check matching
        # ----------------------------------------------------

        if not liked_idxs:

            return jsonify({

                "error":
                    "None of the movies were found.",

                "matched_movies":
                    matched_titles,

                "unmatched_movies":
                    unmatched_titles

            }), 400


        # ----------------------------------------------------
        # User preference vector
        # ----------------------------------------------------

        p_u = Q_norm[
            liked_idxs
        ].mean(
            axis=0
        )


        # Normalize
        p_u = (
            p_u
            /
            (
                np.linalg.norm(
                    p_u
                )
                + 1e-8
            )
        )


        # ----------------------------------------------------
        # Content score (MF)
        # ----------------------------------------------------

        content_score = (
            Q_norm @ p_u
        )


        # ----------------------------------------------------
        # Item-Item CF score
        # ----------------------------------------------------

        if item_item_sim is not None and liked_idxs:
            liked_set = set(liked_idxs)
            sim_subset = item_item_sim[liked_idxs]
            ii_raw = sim_subset.sum(axis=0).A1
            ii_raw[list(liked_set)] = -np.inf
        else:
            ii_raw = np.full(len(Q), -np.inf, dtype=np.float32)

        ii_scores = np.nan_to_num(ii_raw, neginf=-1e9)

        # Normalize II scores to [0, 1] range
        finite_mask = ii_scores > -1e8
        if finite_mask.any():
            ii_min = ii_scores[finite_mask].min()
            ii_max = ii_scores[finite_mask].max()
            if ii_max > ii_min:
                ii_scores_norm = (ii_scores - ii_min) / (ii_max - ii_min)
            else:
                ii_scores_norm = np.zeros_like(ii_scores)
        else:
            ii_scores_norm = np.zeros_like(ii_scores)


        # ----------------------------------------------------
        # Genre score
        # ----------------------------------------------------

        genre_scores = (
            calculate_genre_scores(
                selected_genres
            )
        )


        # ----------------------------------------------------
        # Combine scores
        # ----------------------------------------------------

        if mood == "Something popular":

            mf_component = (
                0.60
                * content_score
                +
                0.20
                * genre_scores
                +
                0.15
                * popularity_score
            )

            final_scores = hybrid_scores(mf_component, ii_scores_norm, alpha=0.65)


        elif mood == "Hidden gem / underrated":

            mf_component = (
                0.70
                * content_score
                +
                0.20
                * genre_scores
                -
                0.05
                * popularity_score
            )

            final_scores = hybrid_scores(mf_component, ii_scores_norm, alpha=0.70)


        else:

            mf_component = (

                0.75
                * content_score

                +

                0.25
                * genre_scores

            )


            final_scores = hybrid_scores(mf_component, ii_scores_norm, alpha=0.75)


        # ----------------------------------------------------
        # Allowed movies
        # ----------------------------------------------------

        allowed_idx = np.ones(
            len(final_scores),
            dtype=bool
        )


        # Apply mood
        allowed_idx = (
            apply_mood_filter(
                mood,
                allowed_idx
            )
        )


        # Remove movies user already entered
        for idx in liked_idxs:

            allowed_idx[idx] = False


        # ----------------------------------------------------
        # Candidate ranking
        # ----------------------------------------------------

        candidate_idx = np.where(
            allowed_idx
        )[0]


        if len(candidate_idx) == 0:

            return jsonify({

                "recommendations": [],

                "matched_movies":
                    matched_titles,

                "unmatched_movies":
                    unmatched_titles,

                "error":
                    "No movies matched the selected filters."

            }), 400


        ranked = candidate_idx[
            np.argsort(
                final_scores[
                    candidate_idx
                ]
            )[::-1]
        ]


        top_idx = ranked[:10]


        # ----------------------------------------------------
        # Build recommendations
        # ----------------------------------------------------

        recommendations = []


        for idx in top_idx:

            idx = int(idx)


            # ------------------------------------------------
            # Movie ID
            # ------------------------------------------------

            movie_id = int(
                idx2movie[idx]
            )


            # ------------------------------------------------
            # Movie information
            # ------------------------------------------------

            if movie_id not in movie_info.index:

                continue


            row = movie_info.loc[
                movie_id
            ]


            title = row["title"]

            genres = row["genres"]


            # ------------------------------------------------
            # Similar movies
            # ------------------------------------------------

            similarities = (
                Q_norm[
                    liked_idxs
                ]
                @
                Q_norm[idx]
            )


            best_positions = np.argsort(
                similarities
            )[::-1][:2]


            similar_titles = []


            for position in best_positions:

                liked_idx = (
                    liked_idxs[
                        int(position)
                    ]
                )


                liked_movie_id = int(
                    idx2movie[
                        liked_idx
                    ]
                )


                if liked_movie_id in movie_info.index:

                    similar_title = (
                        movie_info.loc[
                            liked_movie_id,
                            "title"
                        ]
                    )


                    similar_titles.append(
                        similar_title
                    )


            # ------------------------------------------------
            # Reasons
            # ------------------------------------------------

            reasons = []


            if similar_titles:

                reasons.append(
                    "Similar to "
                    +
                    ", ".join(
                        similar_titles
                    )
                )


            # Genre reason
            if selected_genres:

                movie_genres = set(
                    str(
                        genres
                    ).split("|")
                )


                selected_set = set(
                    selected_genres
                )


                matched_genres = (
                    movie_genres
                    &
                    selected_set
                )


                if matched_genres:

                    reasons.append(
                        "Matches your "
                        +
                        ", ".join(
                            sorted(
                                matched_genres
                            )
                        )
                        +
                        " preference"
                    )


            # Mood reason
            if mood == "Something popular":

                if popularity[idx] > POPULARITY_THRESHOLD:

                    reasons.append(
                        "Popular with viewers"
                    )


            elif mood == "Hidden gem / underrated":

                reasons.append(
                    "Less mainstream pick"
                )


            elif mood == "Recent releases":

                movie_year = (
                    year_by_idx[idx]
                )

                if not np.isnan(
                    movie_year
                ):

                    reasons.append(
                        f"Released in "
                        f"{int(movie_year)}"
                    )


            elif mood == "Classic / older films":

                movie_year = (
                    year_by_idx[idx]
                )

                if not np.isnan(
                    movie_year
                ):

                    reasons.append(
                        f"Classic film from "
                        f"{int(movie_year)}"
                    )


            # ------------------------------------------------
            # Recommendation object
            # (media fetched in parallel after the loop)
            # ------------------------------------------------

            recommendations.append({

                "movieId":
                    movie_id,

                "title":
                    str(title),

                "genres":
                    str(genres),

                "score":
                    round(
                        float(
                            final_scores[idx]
                        ),
                        4
                    ),

                "poster":
                    None,

                "trailer":
                    None,

                "reasons":
                    reasons

            })


        # ----------------------------------------------------
        # Fetch posters + trailers in parallel
        # ----------------------------------------------------

        from concurrent.futures import ThreadPoolExecutor

        titles = [rec["title"] for rec in recommendations]


        def _fetch_media(t):

            try:

                return get_movie_media(t)

            except Exception:

                return {"poster": None, "trailer": None}


        if titles:

            with ThreadPoolExecutor(max_workers=10) as pool:

                media_results = list(
                    pool.map(_fetch_media, titles)
                )


            for rec, media in zip(
                recommendations,
                media_results
            ):

                rec["poster"] = media.get("poster")

                rec["trailer"] = media.get("trailer")


        # ----------------------------------------------------
        # Return response
        # ----------------------------------------------------

        return jsonify({

            "recommendations":
                recommendations,

            "matched_movies":
                matched_titles,

            "unmatched_movies":
                unmatched_titles

        })


    # ========================================================
    # ERROR HANDLING
    # ========================================================

    except Exception:

        print(
            "\n"
            "========================================"
        )

        print(
            "ERROR IN /recommend"
        )

        print(
            "========================================"
        )

        traceback.print_exc()

        print(
            "========================================\n"
        )


        return jsonify({

            "error":
                "Something went wrong while building "
                "recommendations. Please try again."

        }), 500


# ============================================================
# START SERVER
# ============================================================

if __name__ == "__main__":

    print(
        "\n"
        "========================================"
    )

    print(
        "CineMatch Recommendation API"
    )

    print(
        "========================================"
    )

    print(
        "Model movies:",
        len(Q)
    )

    print(
        "Latent factors:",
        Q.shape[1]
    )

    print(
        "Server:"
    )

    print(
        f"http://{FLASK_HOST}:{FLASK_PORT}"
    )

    print(
        f"Debug mode: {FLASK_DEBUG}"
    )

    print(
        "========================================\n"
    )


    app.run(
        host=FLASK_HOST,
        port=FLASK_PORT,
        debug=FLASK_DEBUG
    )