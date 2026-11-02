from pathlib import Path

import numpy as np
import pandas as pd
import requests
import os
import re
import time
from urllib.parse import quote
from scipy.sparse import csr_matrix

from dotenv import load_dotenv


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MOVIES_PATH = PROJECT_ROOT / "data" / "movies.csv"
MODEL_PATH = PROJECT_ROOT / "model_weights.npz"
POPULARITY_PATH = PROJECT_ROOT / "movie_popularity.npy"
ENV_PATH = PROJECT_ROOT / ".env"


# ============================================================
# ENVIRONMENT
# ============================================================

load_dotenv(ENV_PATH)

TMDB_API_KEY = os.getenv("TMDB_API_KEY")


if TMDB_API_KEY:

    print("TMDB API key loaded successfully.")

else:

    print(
        "WARNING: TMDB_API_KEY not found."
    )


# ============================================================
# TMDB SETTINGS
# ============================================================

TMDB_IMAGE_BASE = (
    "https://image.tmdb.org/t/p/"
)

TMDB_POSTER_SIZE = "w500"
TMDB_BACKDROP_SIZE = "w780"


# ============================================================
# CLEAN MOVIE TITLE
# ============================================================

def clean_movie_title(title):

    if not title:

        return ""

    title = str(title).strip()

    # Remove MovieLens year
    title = re.sub(
        r"\s*\(\d{4}\)\s*$",
        "",
        title
    ).strip()

    # Convert:
    #
    # Avengers, The
    #
    # into:
    #
    # The Avengers

    match = re.match(
        r"^(.*),\s*(the|a|an)$",
        title,
        flags=re.IGNORECASE
    )

    if match:

        title = (
            f"{match.group(2)} "
            f"{match.group(1)}"
        )

    return title.strip()


# ============================================================
# EXTRACT YEAR
# ============================================================

def extract_movie_year(title):

    if not title:

        return None

    match = re.search(
        r"\((\d{4})\)\s*$",
        str(title)
    )

    if match:

        return int(
            match.group(1)
        )

    return None


# ============================================================
# NORMALIZE TITLE FOR COMPARISON
# ============================================================

def normalize_title(title):

    if not title:

        return ""

    title = clean_movie_title(title)

    title = title.lower()

    title = re.sub(
        r"[^a-z0-9\s]",
        " ",
        title
    )

    title = re.sub(
        r"\s+",
        " ",
        title
    ).strip()

    return title


# ============================================================
# CALCULATE TMDB MATCH SCORE
# ============================================================

def calculate_tmdb_match_score(
    result,
    requested_title,
    requested_year
):

    result_title = result.get(
        "title",
        ""
    )

    result_original_title = result.get(
        "original_title",
        ""
    )

    requested_normalized = normalize_title(
        requested_title
    )

    result_normalized = normalize_title(
        result_title
    )

    original_normalized = normalize_title(
        result_original_title
    )

    score = 0

    # --------------------------------------------------------
    # TITLE MATCH
    # --------------------------------------------------------

    if result_normalized == requested_normalized:

        score += 100

    elif original_normalized == requested_normalized:

        score += 95

    elif requested_normalized in result_normalized:

        score += 60

    elif result_normalized in requested_normalized:

        score += 50

    # --------------------------------------------------------
    # YEAR MATCH
    # --------------------------------------------------------

    release_date = result.get(
        "release_date",
        ""
    )

    result_year = None

    if release_date:

        try:

            result_year = int(
                release_date[:4]
            )

        except (ValueError, TypeError):

            result_year = None

    if requested_year and result_year:

        if result_year == requested_year:

            score += 80

        elif abs(
            result_year - requested_year
        ) == 1:

            score += 20

    # --------------------------------------------------------
    # IMAGE BONUS
    # --------------------------------------------------------

    if result.get("poster_path"):

        score += 30

    elif result.get("backdrop_path"):

        score += 10

    # --------------------------------------------------------
    # POPULARITY
    # --------------------------------------------------------

    popularity = result.get(
        "popularity",
        0
    )

    try:

        score += min(
            float(popularity) / 10,
            10
        )

    except (ValueError, TypeError):

        pass

    return score


# ============================================================
# YOUTUBE FALLBACK
# ============================================================

def get_youtube_search_url(title):

    clean_title = clean_movie_title(
        title
    )

    query = (
        f"{clean_title} official trailer"
    )

    return (
        "https://www.youtube.com/results?search_query="
        + quote(query)
    )


# ============================================================
# SMALL TTL CACHE
# ============================================================
#
# Stable lookups ("this movie really has no poster") are kept
# forever, while transient network failures are kept only
# briefly so they are retried instead of being remembered for
# the lifetime of the process.
#

FAILURE_TTL_SECONDS = 300


def _cache_put(cache, key, value, stable):

    expires = (
        None
        if stable
        else time.time() + FAILURE_TTL_SECONDS
    )

    cache[key] = (value, expires)


def _cache_get(cache, key):

    entry = cache.get(key)

    if entry is None:

        return None, False

    value, expires = entry

    if expires is not None and time.time() > expires:

        cache.pop(key, None)

        return None, False

    return value, True


# ============================================================
# TMDB MOVIE MEDIA
# ============================================================

_movie_media_cache = {}


def get_movie_media(title):

    """
    Get poster and YouTube trailer from TMDB.

    Poster priority:

        1. TMDB poster
        2. TMDB backdrop
        3. Generated fallback poster

    Trailer priority:

        1. Official YouTube trailer
        2. Any YouTube trailer
        3. YouTube search fallback

    Returns:

        {
            "poster": str,
            "trailer": str,
            "poster_type": str
        }
    """

    if not title:

        return {
            "poster": None,
            "trailer": None,
            "poster_type": "none"
        }


    cached, found = _cache_get(
        _movie_media_cache,
        title
    )

    if found:

        return cached


    media, stable = _fetch_movie_media(title)

    _cache_put(
        _movie_media_cache,
        title,
        media,
        stable
    )


    return media


def _fetch_movie_media(title):

    """
    Resolve TMDB media for a title.

    Returns (media, stable) where stable is False for
    transient failures, which are only cached briefly.
    """

    if not title:

        return ({
            "poster": None,
            "trailer": None,
            "poster_type": "none"
        }, True)


    # --------------------------------------------------------
    # API KEY CHECK
    # --------------------------------------------------------

    if not TMDB_API_KEY:

        print(
            "TMDB API key missing."
        )

        return ({
            "poster": create_fallback_poster(
                title
            ),
            "trailer": get_youtube_search_url(
                title
            ),
            "poster_type": "fallback"
        }, True)


    clean_title = clean_movie_title(
        title
    )

    requested_year = extract_movie_year(
        title
    )


    # ========================================================
    # SEARCH TMDB
    # ========================================================

    try:

        search_url = (
            "https://api.themoviedb.org/3/"
            "search/movie"
        )

        search_params = {

            "api_key":
                TMDB_API_KEY,

            "query":
                clean_title,

            "language":
                "en-US",

            "include_adult":
                False,

            "page":
                1
        }


        # ----------------------------------------------------
        # Include year when available
        # ----------------------------------------------------

        if requested_year:

            search_params[
                "year"
            ] = requested_year


        response = requests.get(
            search_url,
            params=search_params,
            timeout=8
        )


        if response.status_code != 200:

            print(
                "TMDB search error:",
                response.status_code,
                clean_title
            )

            # Transient failure: do not cache.
            return (
                build_fallback_media(title),
                False
            )


        results = response.json().get(
            "results",
            []
        )


        # ----------------------------------------------------
        # If year search gives nothing,
        # retry without year.
        # ----------------------------------------------------

        if not results and requested_year:

            search_params.pop(
                "year",
                None
            )

            response = requests.get(
                search_url,
                params=search_params,
                timeout=8
            )

            if response.status_code == 200:

                results = response.json().get(
                    "results",
                    []
                )

            else:

                # Transient failure: do not cache.
                return (
                    build_fallback_media(title),
                    False
                )


        if not results:

            print(
                "TMDB movie not found:",
                clean_title
            )

            # Stable fact: this title has no TMDB match.
            return (
                build_fallback_media(title),
                True
            )


        # ====================================================
        # RANK RESULTS
        # ====================================================

        ranked_results = sorted(

            results,

            key=lambda result:
                calculate_tmdb_match_score(
                    result,
                    title,
                    requested_year
                ),

            reverse=True
        )


        # ====================================================
        # FIND BEST RESULT WITH POSTER
        # ====================================================

        movie = None

        for result in ranked_results:

            if result.get(
                "poster_path"
            ):

                movie = result

                break


        # ----------------------------------------------------
        # If no result has poster,
        # use best matching result.
        # ----------------------------------------------------

        if movie is None:

            movie = ranked_results[0]


        tmdb_id = movie.get(
            "id"
        )


        # ====================================================
        # POSTER
        # ====================================================

        poster = None

        poster_type = "none"


        poster_path = movie.get(
            "poster_path"
        )


        if poster_path:

            poster = (
                TMDB_IMAGE_BASE
                + TMDB_POSTER_SIZE
                + poster_path
            )

            poster_type = "tmdb"


        # ====================================================
        # BACKDROP FALLBACK
        # ====================================================

        if poster is None:

            backdrop_path = movie.get(
                "backdrop_path"
            )

            if backdrop_path:

                poster = (
                    TMDB_IMAGE_BASE
                    + TMDB_BACKDROP_SIZE
                    + backdrop_path
                )

                poster_type = "backdrop"


        # ====================================================
        # TRAILER
        # ====================================================

        trailer = None


        if tmdb_id:

            trailer = get_tmdb_trailer(
                tmdb_id
            )


        # ----------------------------------------------------
        # YouTube fallback
        # ----------------------------------------------------

        if trailer is None:

            trailer = get_youtube_search_url(
                title
            )


        # ====================================================
        # FINAL POSTER FALLBACK
        # ====================================================

        if poster is None:

            poster = create_fallback_poster(
                title
            )

            poster_type = "fallback"


        return ({

            "poster":
                poster,

            "trailer":
                trailer,

            "poster_type":
                poster_type
        }, True)


    except requests.RequestException as e:

        print(
            "TMDB request error:",
            e
        )

        # Transient failure: do not cache.
        return (
            build_fallback_media(title),
            False
        )


    except Exception as e:

        print(
            "TMDB media error:",
            e
        )

        # Transient failure: do not cache.
        return (
            build_fallback_media(title),
            False
        )


# ============================================================
# TMDB TRAILER
# ============================================================

_trailer_cache = {}


def get_tmdb_trailer(tmdb_id):

    """
    Resolve a YouTube trailer URL for a TMDB movie id.

    Transient failures are only cached briefly so they are
    retried later.
    """

    if not tmdb_id:

        return None


    cached, found = _cache_get(
        _trailer_cache,
        tmdb_id
    )

    if found:

        return cached


    trailer, stable = _fetch_tmdb_trailer(tmdb_id)

    _cache_put(
        _trailer_cache,
        tmdb_id,
        trailer,
        stable
    )


    return trailer


def _fetch_tmdb_trailer(tmdb_id):

    """Returns (trailer_url_or_None, stable)."""

    try:

        video_url = (
            "https://api.themoviedb.org/3/"
            f"movie/{tmdb_id}/videos"
        )

        video_params = {

            "api_key":
                TMDB_API_KEY,

            "language":
                "en-US"
        }


        response = requests.get(

            video_url,

            params=video_params,

            timeout=8
        )


        if response.status_code != 200:

            # Transient failure: do not cache.
            return (None, False)


        videos = response.json().get(
            "results",
            []
        )


        # ====================================================
        # ONLY YOUTUBE
        # ====================================================

        youtube_videos = [

            video

            for video in videos

            if video.get(
                "site"
            ) == "YouTube"

        ]


        if not youtube_videos:

            # Stable fact: no YouTube videos on TMDB.
            return (None, True)


        # ====================================================
        # OFFICIAL TRAILERS
        # ====================================================

        official_trailers = [

            video

            for video in youtube_videos

            if (

                video.get(
                    "type"
                ) == "Trailer"

                and

                video.get(
                    "official"
                ) is True

            )

        ]


        # ====================================================
        # ANY TRAILERS
        # ====================================================

        trailers = [

            video

            for video in youtube_videos

            if video.get(
                "type"
            ) == "Trailer"

        ]


        # ====================================================
        # TEASER FALLBACK
        # ====================================================

        teasers = [

            video

            for video in youtube_videos

            if video.get(
                "type"
            ) == "Teaser"

        ]


        selected_video = None


        if official_trailers:

            selected_video = (
                official_trailers[0]
            )

        elif trailers:

            selected_video = (
                trailers[0]
            )

        elif teasers:

            selected_video = (
                teasers[0]
            )


        if selected_video:

            key = selected_video.get(
                "key"
            )

            if key:

                return (
                    "https://www.youtube.com/watch?v="
                    + key,
                    True
                )


        return (None, True)


    except Exception as e:

        print(
            "TMDB trailer error:",
            e
        )

        # Transient failure: do not cache.
        return (None, False)


# ============================================================
# FALLBACK MEDIA
# ============================================================

def build_fallback_media(title):

    return {

        "poster":
            create_fallback_poster(
                title
            ),

        "trailer":
            get_youtube_search_url(
                title
            ),

        "poster_type":
            "fallback"

    }


# ============================================================
# FALLBACK POSTER
# ============================================================

def create_fallback_poster(title):

    """
    Creates a clean poster-style fallback image
    using an external placeholder service.

    This means every recommendation still
    has something that looks like a poster.
    """

    clean_title = clean_movie_title(
        title
    )

    encoded_title = quote(
        clean_title
    )

    return (
        "https://placehold.co/"
        "500x750/"
        "11131a/"
        "ffffff"
        "?text="
        + encoded_title
    )


# ============================================================
# LOAD MOVIES
# ============================================================

print(
    "Loading movie data..."
)

movies = pd.read_csv(
    MOVIES_PATH
)

print(
    f"Loaded {len(movies):,} movies"
)


# ============================================================
# LOAD TRAINED MODEL
# ============================================================

print(
    "Loading trained model..."
)

with np.load(
    MODEL_PATH
) as weights:

    Q = weights["Q"]

    b_i = weights["b_i"]

    global_mean = float(
        weights["global_mean"]
    )

    movie_ids = weights[
        "movie_ids"
    ]


print(
    f"Loaded Q matrix: {Q.shape}"
)

print(
    f"Loaded movie IDs: "
    f"{len(movie_ids):,}"
)


# ============================================================
# LOAD POPULARITY
# ============================================================

print(
    "Loading movie popularity..."
)

popularity = np.load(
    POPULARITY_PATH
)

print(
    f"Loaded popularity: "
    f"{len(popularity):,}"
)


# ============================================================
# VALIDATE MODEL DATA
# ============================================================

if len(Q) != len(movie_ids):

    raise ValueError(
        "Q matrix and movie_ids "
        "have different lengths."
    )


if len(Q) != len(popularity):

    raise ValueError(
        "Q matrix and popularity "
        "have different lengths."
    )


# ============================================================
# LOAD ITEM-ITEM SIMILARITY
# ============================================================

ITEM_ITEM_SIM_PATH = PROJECT_ROOT / "item_item_sim.npz"

print("Loading item-item similarity...")

if ITEM_ITEM_SIM_PATH.exists():
    with np.load(ITEM_ITEM_SIM_PATH) as data:
        item_item_sim = csr_matrix(
            (data['data'], data['indices'], data['indptr']),
            shape=data['shape']
        )
    print(f"Loaded item-item similarity: {item_item_sim.shape}, {item_item_sim.nnz:,} non-zero")
else:
    print("WARNING: item_item_sim.npz not found. Run src/item_item_cf.py first.")
    item_item_sim = None


# ============================================================
# MOVIE ID <-> MODEL INDEX
# ============================================================

movie2idx = {

    int(movie_id): i

    for i, movie_id
    in enumerate(movie_ids)

}


idx2movie = {

    i: int(movie_id)

    for i, movie_id
    in enumerate(movie_ids)

}


# ============================================================
# ALIGN MOVIES WITH MODEL
# ============================================================

model_movies = movies[
    movies["movieId"].isin(
        movie_ids
    )
].copy()


model_movies["model_idx"] = (

    model_movies[
        "movieId"
    ]

    .map(
        movie2idx
    )

)


model_movies = (

    model_movies

    .dropna(
        subset=[
            "model_idx"
        ]
    )

    .sort_values(
        "model_idx"
    )

    .reset_index(
        drop=True
    )

)


# ============================================================
# CHECK ALIGNMENT
# ============================================================

if len(model_movies) != len(Q):

    print(
        "WARNING:"
    )

    print(
        f"Model movies: "
        f"{len(model_movies):,}"
    )

    print(
        f"Q rows: "
        f"{len(Q):,}"
    )

else:

    print(
        "Movie/model alignment OK."
    )


# ============================================================
# NORMALIZE Q
# ============================================================

Q_norm = Q / (

    np.linalg.norm(
        Q,
        axis=1,
        keepdims=True
    )

    + 1e-8

)


# ============================================================
# MOVIE LOOKUP
# ============================================================

movie_info = movies.set_index(
    "movieId"
)


# ============================================================
# FINISHED
# ============================================================

print(
    "Recommendation model loaded successfully!"
)

print(
    f"Model contains "
    f"{len(movie2idx):,} movies"
)

print(
    f"Latent factors: "
    f"{Q.shape[1]}"
)