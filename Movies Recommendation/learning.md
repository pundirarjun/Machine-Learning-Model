# CineMatch — Movie Recommendation System

Everything you need to understand, run and extend this project: the machine
learning model, the data, the API, the frontend, the evaluation methodology and
the known limitations.

---

## 1. What this project is

CineMatch is a full-stack movie recommender:

1. The user types a few movies they enjoyed, picks genres and a mood.
2. A **matrix factorization model** (trained on 32 million MovieLens ratings)
   maps those movies into a 20-dimensional "taste space".
3. The backend blends **latent similarity + genre match + popularity**, applies
   mood filters and returns the top 10 movies with human-readable reasons.
4. Posters and trailers are resolved through the **TMDB API** (with graceful
   fallbacks).
5. A **React + Vite** frontend renders the results.

```
┌──────────────────────┐   POST /recommend   ┌─────────────────────────────┐
│  React (Vite)        │ ──────────────────► │  Flask API (backend/app.py) │
│  frontend/src/App.jsx│ ◄────────────────── │                             │
└──────────────────────┘   JSON results      │  ┌───────────────────────┐  │
                                             │  │ recommender.py        │  │
        TMDB API ◄───────────────────────────┤  │  • model + data load  │  │
   (posters / trailers)                     │  │  • title matching     │  │
                                             │  │  • TMDB media lookup  │  │
                                             │  └───────────────────────┘  │
                                             │  ┌───────────────────────┐  │
                                             │  │ model_weights.npz     │  │
                                             │  │  Q (84432×20), b_i, … │  │
                                             │  └───────────────────────┘  │
                                             └─────────────────────────────┘

        offline training pipeline (src/)
   data/ratings.csv ──► data_loader.py ──► train.py ──► model_weights.npz
                                                     ──► movie_popularity.npy
                                                     ──► training_history.npz
                                                     ──► plot_training.py / evaluate_recommender.py
```

---

## 2. Repository layout

```
Movies Recommendation/
├── data/                       MovieLens ml-32m (ratings.csv is 836 MB, on Git LFS)
│   ├── movies.csv              87,585 titles + genres
│   ├── ratings.csv             32,000,204 ratings        (Git LFS)
│   ├── tags.csv                2,000,072 tags            (Git LFS)
│   ├── links.csv               movieId → imdbId / tmdbId
│   └── README.txt, checksums.txt
│
├── src/                        OFFLINE ML PIPELINE (run from anywhere)
│   ├── data_loader.py          CSV → pandas + user/item index maps
│   ├── model.py                MatrixFactorization + Numba SGD kernels
│   ├── train.py                end-to-end training script
│   ├── tune_hyperparams.py     resumable hyperparameter sweep
│   ├── show_history.py         print the last training run's numbers
│   ├── evaluate_recommender.py Precision@10 / Recall@10 / NDCG@10
│   └── plot_training.py        train vs val RMSE curve
│
├── backend/                    ONLINE SERVING
│   ├── recommender.py          loads model/data, title matching, TMDB media
│   └── app.py                  Flask routes, scoring, filters, response build
│
├── frontend/                   REACT UI (Vite)
│   └── src/App.jsx             single-page app (form → API → results grid)
│
├── model_weights.npz           trained artifacts   (Git LFS)
├── movie_popularity.npy        rating count per model movie
├── training_history.npz        per-epoch train/val RMSE + learning rate
├── tune_results.json           hyperparameter sweep results (resumable)
├── Training vs Validation RMSE plot.png
├── requirements.txt            Python dependencies
├── .env / .env.example         TMDB_API_KEY (`.env` is git-ignored)
└── learning.md                 this file
```

---

## 3. Dataset — MovieLens 32M (ml-32m)

| Property | Value |
|---|---|
| Ratings | 32,000,204 |
| Users | 200,948 (each rated ≥ 20 movies) |
| Movies in `movies.csv` | 87,585 |
| Movies with ≥ 1 rating | **84,432** (this is what the model knows) |
| Rating scale | 0.5 – 5.0 stars, half-star steps |
| Period | 1995-01-09 → 2023-10-12 |
| Files | `ratings.csv`, `movies.csv`, `tags.csv`, `links.csv` |

Important consequence: **3,153 movies have no ratings and can never be
recommended** (they are filtered out by `data_loader.load_data()`, which only
builds indices for movie IDs present in `ratings.csv`).

Licence: GroupLens research licence — non-commercial, must cite
Harper & Konstan, *The MovieLens Datasets: History and Context*, TiiS 2015.

---

## 4. The machine learning model

### 4.1 Problem framing

Given the sparse user×movie rating matrix, learn low-dimensional vectors such
that

```
predicted_rating(user u, movie i) ≈ actual rating
```

This is a **latent factor model (matrix factorization) with biases** — the
classic "Simon Funk / Netflix Prize" approach.

### 4.2 Prediction equation

```
r̂(u,i) = μ  +  b_u[u]  +  b_i[i]  +  P[u] · Q[i]
          │      │          │            └─ dot product of 20-dim latent vectors
          │      │          └─ movie bias  (is this movie generally liked?)
          │      └─ user bias (is this user generous/critical?)
          └─ global mean rating = 3.5404
```

Where:

- `P` — user factors, shape `(200948, 20)`, `float32`
- `Q` — movie factors, shape `(84432, 20)`, `float32`
- `b_u` — user biases, shape `(200948,)`
- `b_i` — movie biases, shape `(84432,)`
- `μ` — mean of the **training** ratings (3.5404 on the full dataset)

### 4.3 Training objective

```
L = Σ (r_ui − r̂_ui)²  +  λ · ( ‖P_u‖² + ‖Q_i‖² + b_u² + b_i² )
```

Minimised by **stochastic gradient descent**, one rating at a time.

### 4.4 SGD update rules (src/model.py:sgd_epoch)

For each sampled rating `(u, i, r)`:

```
err = r − ( μ + b_u[u] + b_i[i] + P[u]·Q[i] )

b_u[u] ← b_u[u] + lr · ( err − λ·b_u[u] )
b_i[i] ← b_i[i] + lr · ( err − λ·b_i[i] )

for each factor f:
    P[u,f] ← P[u,f] + lr · ( err·Q[i,f] − λ·P[u,f] )
    Q[i,f] ← Q[i,f] + lr · ( err·P[u,f] − λ·Q[i,f] )   ← uses the OLD P[u,f]
```

`err` is computed **before** any update, so the Q update still sees the
pre-update `P` — this is the standard simultaneous-update formulation.

### 4.5 Why Numba?

`sgd_epoch` runs 25,600,163 iterations per epoch (80 % of 32,000,204) × up to
25 epochs ≈ 640 million updates. In pure Python that would take hours.
`@njit` compiles the loop to machine code — roughly 20–100× faster — while
keeping the code readable.

> ⚠️ **Numba runs without bounds checking.** An out-of-range user/item index
> does not raise `IndexError`; it silently reads/writes random memory and
> corrupts the process (crashes, NaN weights, LLVM assertion failures).
> `MatrixFactorization.fit()` now validates indices and array lengths up front
> for exactly this reason.

### 4.6 Hyperparameters

| Hyperparameter | Value | Where |
|---|---|---|
| Latent factors `k` | 20 | `src/train.py` (`N_FACTORS`) |
| Base learning rate `lr` | 0.01 | `src/train.py` (`BASE_LR`) |
| LR decay (per epoch) | 0.95 exponential, floor 1e-4 | `src/train.py` (`LR_DECAY`, `MIN_LR`) |
| Regularization `λ` | 0.02 | `src/train.py` (`REG`) |
| Epochs (max) | 60 | `src/train.py` (`EPOCHS`) |
| Validation split | 0.2 (random, per-rating) | `src/train.py` (`VALIDATION_SPLIT`) |
| Early-stopping patience | 5 epochs | `src/train.py` (`PATIENCE`) |
| Weight init | `N(0, 0.1)` for P and Q, `0` for biases | `src/model.py` |
| Biases init | 0 | `src/model.py` |
| Random seed | 42 | `src/train.py` (`np.random.seed`) |
| Dtype | `float32` everywhere | `src/model.py` |

All knobs live as named constants at the top of `src/train.py`; the sweep in
`src/tune_hyperparams.py` varies the same values.

### 4.7 Training loop semantics (`MatrixFactorization.fit`)

1. Shuffle all rating indices, split 80/20 → `train_order`, `val_order`.
2. `μ` is computed **only from training rows** (so validation stays unseen).
3. Each epoch:
   - `epoch_lr = max(lr · lr_decay**epoch, min_lr)` — exponential decay
   - shuffle `train_order`, run `sgd_epoch` with `epoch_lr` → train RMSE
   - run `validation_rmse` on `val_order` (no updates)
   - append train/val RMSE **and `epoch_lr`** to history
   - if val RMSE improved → snapshot best weights, reset patience counter
   - else increment the no-improvement counter; when it reaches `patience`
     → early stop
4. Restore the best snapshot (not the last epoch).

`train.py` writes `model_weights.npz` / `training_history.npz` through a
temporary file + `os.replace()`, so an interrupted run can never leave a
half-written artifact behind.

Safety guards now in place:

| Condition | Behaviour |
|---|---|
| `validation_split = 0` | no early stopping, final epoch kept, val history = `nan` |
| `validation_split ≥ 1` or `< 0` | `ValueError` |
| empty training slice | `ValueError` |
| index out of range / mismatched lengths | `ValueError` (prevents memory corruption) |
| all val RMSE = NaN | `RuntimeError` instead of `None` weights |

### 4.8 Saved artifacts

`model_weights.npz` (Git LFS, 24 MB):

| Key | Shape | Meaning |
|---|---|---|
| `P` | (200948, 20) float32 | user latent factors |
| `Q` | (84432, 20) float32 | movie latent factors |
| `b_u` | (200948,) float32 | user biases |
| `b_i` | (84432,) float32 | movie biases |
| `global_mean` | scalar float64 | μ |
| `movie_ids` | (84432,) int32 | row `k` of Q ↔ MovieLens `movieId` |

`movie_popularity.npy` — `(84432,)` int32 rating counts, **in the same order as
Q** (built by `train.py`, required by the backend).

`training_history.npz` — `train_rmse`, `val_rmse` and `lr` arrays (one entry
per epoch).

---

## 5. Training results

From `training_history.npz` (seed 42, 60 epochs, LR decay 0.95, patience 5 —
early stopping never fired because every epoch improved slightly):

| Epoch | Train RMSE | Val RMSE | LR |
|---|---|---|---|
| 1 | 0.8896 | 0.8678 | 0.01000 |
| 5 | 0.8026 | 0.8095 | 0.00815 |
| 10 | 0.7519 | 0.7816 | 0.00630 |
| 15 | 0.7278 | 0.7721 | 0.00488 |
| 20 | 0.7143 | 0.7681 | 0.00377 |
| 30 | 0.7003 | 0.7651 | 0.00226 |
| 40 | 0.6936 | 0.7641 | 0.00135 |
| 50 | 0.6900 | 0.7637 | 0.00081 |
| 60 | **0.6880** | **0.7635** | 0.00048 |

Reading the curve (`Training vs Validation RMSE plot.png`):

- The old 25-epoch run ended at val **0.7661** with the curve still falling;
  extending to 60 epochs + exponential LR decay reaches **0.7635**.
- The gap between train and val stays ~0.075 — no sign of overfitting, so
  the remaining error is capacity/regularization, not epoch count.
- Epochs 40→60 buy only ~0.0006 RMSE: the run is now near its plateau at
  `k = 20`.
- For reference, always predicting the mean (μ = 3.5404) gives
  **RMSE = 1.059** (the std of all 32 M ratings).

### 5.1 Hyperparameter sweep (`src/tune_hyperparams.py`)

Five configurations, identical seed → identical 80/20 split, early stopping
disabled (60 epochs for all), so the numbers are directly comparable:

| # | Config | Best val RMSE | Final train | Best epoch | Time |
|---|---|---|---|---|---|
| 1 | **k20, λ0.02, decay 0.95 (current)** | **0.7632** | 0.6877 | 60 | 813 s |
| 2 | k20, λ0.02, constant LR | 0.7660 | 0.6736 | 24 | 1009 s |
| 3 | k40, λ0.05, decay 0.95 | 0.7774 | 0.7355 | 60 | 1162 s |
| 4 | k20, λ0.05, decay 0.95 | 0.7804 | 0.7436 | 60 | 836 s |
| 5 | k20, λ0.10, decay 0.95 | 0.8258 | 0.8122 | 60 | 799 s |

Conclusions — the shipped configuration stays `k20 / λ0.02 / decay 0.95`:

- **Decay beats constant LR** (0.7632 vs 0.7660): without decay the model
  peaks at epoch 24 and then overfits (train drops to 0.6736 while val rises).
- **λ0.02 is the sweet spot** among those tried; λ0.05 and λ0.10 under-fit
  badly (0.78 / 0.83).
- **k40 does not help** at λ0.05 (0.7774) and costs ~40 % more time — extra
  capacity needs stronger regularization to pay off, untested.

Results persist to `tune_results.json` after every config, so
`python src/tune_hyperparams.py` resumes after a crash/interrupt instead of
retraining the finished ones (`--reset` starts over).

---

## 6. Serving — how a recommendation is produced

The serving path never uses the *user* side of the model (`P`, `b_u`,
`μ + b_u`): there is no logged-in user, so this is a **cold-start / content
profile** approach built on the *movie* side.

### 6.1 Steps (`backend/app.py` → `POST /recommend`)

1. **Resolve titles** — every input string goes through `find_movie_match()`:
   - normalise: lowercase → strip `(YYYY)` → flip `"Avengers, The"` →
     `"The Avengers"` → drop punctuation → squeeze spaces
   - exact normalized match (prefer movies that exist in the model, tie-break by
     popularity)
   - otherwise substring match, scored `100 / 80 / 60 / 20` by match quality +
     `min(popularity/1000, 10)` as a popularity bonus
   - unmatched titles are reported back, not fatal
2. **Build a taste vector** from the matched movies:

   ```
   p_u = normalize( mean( normalize(Q[i]) for i in liked ) )
   ```
3. **Content score** — cosine similarity against every movie:
   `content = Q_norm @ p_u` ∈ [-1, 1]
4. **Genre score** — per movie: `(# matched genres) / (# requested genres)` ∈ [0, 1]
   (computed with precomputed per-genre boolean masks over 84,432 movies).
5. **Popularity score** — `count / max(count)` ∈ [0, 1].
6. **Blend** (weights chosen by hand, not tuned):

   | Mood | Formula |
   |---|---|
   | Something popular | `0.65·content + 0.20·genre + 0.15·popularity` |
   | Hidden gem / underrated | `0.75·content + 0.20·genre − 0.05·popularity` |
   | Recent releases | `0.75·content + 0.25·genre` |
   | Classic / older films | `0.75·content + 0.25·genre` |

7. **Filters** — start from all model movies, then:
   - Recent → keep `year ≥ 2015`
   - Classic → keep `year < 1990`
   - Hidden gem → keep `popularity < 70th percentile`
   - always drop the movies the user typed in
8. **Rank & take top 10**, then attach reasons, poster and trailer.

### 6.2 Reasons

- *"Similar to <your movie>"* — the 2 most similar of the movies you entered
- *"Matches your Action preference"* — genre intersection
- *"Popular with viewers"* — above the 70th popularity percentile
- *"Released in 2019" / "Classic film from 1984"* — from the title's `(YYYY)`

### 6.3 TMDB media pipeline (`backend/recommender.py`)

```
get_movie_media(title)
  ├─ cache hit? → return (stable entries forever, failures 5 min TTL)
  └─ search TMDB /search/movie (query + year, retry without year)
       ├─ rank results: title match (100/95/60/50)
       │                + year match (80 exact / 20 ±1)
       │                + has poster (30) / backdrop (10)
       │                + popularity (≤10)
       ├─ poster: TMDB poster → TMDB backdrop → placehold.co placeholder
       ├─ trailer: /movie/{id}/videos → official Trailer → any Trailer →
       │           Teaser → YouTube search URL
       └─ failures (timeouts, 5xx) are cached for 5 minutes only, so a bad
          network call does not poison the cache forever
```

If `TMDB_API_KEY` is missing, every movie gets a placeholder poster and a
YouTube **search** link — the app still works.

---

## 7. API reference

Base URL: `http://127.0.0.1:5000` (override with `FLASK_HOST`, `FLASK_PORT`,
`FLASK_DEBUG`).

### `GET /`
```json
{ "message": "Movie Recommendation API is running!", "movies": 84432 }
```

### `GET /health`
```json
{ "status": "ok", "model_movies": 84432, "dataset_movies": 87585, "model_factors": 20 }
```

### `POST /recommend`

Request:
```json
{
  "movies": ["Iron Man (2008)"],
  "genres": ["Action"],
  "mood": "Something popular"
}
```

`mood` must be one of: `Something popular`, `Hidden gem / underrated`,
`Recent releases`, `Classic / older films`.

Success — `200`:
```json
{
  "recommendations": [
    {
      "movieId": 89745,
      "title": "Avengers, The (2012)",
      "genres": "Action|Adventure|Sci-Fi|IMAX",
      "score": 0.8587,
      "poster": "https://image.tmdb.org/t/p/w500/…",
      "trailer": "https://www.youtube.com/watch?v=…",
      "reasons": [
        "Similar to Iron Man (2008)",
        "Matches your Action preference",
        "Popular with viewers"
      ]
    }
  ],
  "matched_movies": ["Iron Man (2008)"],
  "unmatched_movies": []
}
```

Errors — `400` (message in `error`):

| Case | Message |
|---|---|
| empty/missing body | `Request body is required.` |
| `movies` not a list | `movies must be a list.` |
| no titles | `Please provide at least one movie.` |
| > 50 titles | `Please provide at most 50 movies.` |
| `genres` not a list | `genres must be a list.` |
| unknown mood | `Unknown mood.` |
| nothing matched | `None of the movies were found.` (also returns `matched_movies`, `unmatched_movies`) |
| filters removed everything | `No movies matched the selected filters.` |

`500` → `Something went wrong while building recommendations. Please try again.`
(details are logged server-side only).

---

## 8. Frontend (`frontend/src/App.jsx`)

- Single component, React state: `selectedGenres`, `mood`, `movies`,
  `recommendations`, `matchedMovies`, `unmatchedMovies`, `loading`, `error`.
- API base URL: `import.meta.env.VITE_API_URL` (fallback `http://127.0.0.1:5000`).
- Request is aborted after **30 s** (`AbortController`) → "The request timed
  out." instead of an infinite spinner.
- `data.error` is surfaced **regardless of HTTP status** (the backend uses
  400 for validation and also returns an `error` key on some 200s).
- Previous results are cleared when a new request starts.
- Posters: `onError` swaps to a placeholder (and never loops), first 4 images
  load eagerly, the rest lazy.
- Trailer opens in a new tab with `noopener,noreferrer`.
- `npm run lint` (ESLint flat config) and `npm run build` both pass.

---

## 9. Evaluation (`src/evaluate_recommender.py`)

Metrics (top-10, "relevant" = test rating ≥ 4.0, users with ≥ 20 ratings,
1000 random users, seed 42):

- **Precision@10** = |relevant ∩ top-10| / 10
- **Recall@10** = |relevant ∩ top-10| / |relevant|
- **NDCG@10** = DCG@10 / IDCG@10 with `gain = 1/log2(rank+1)`

Current result (60-epoch model):

```
Users evaluated: 988
Precision@10: 0.0355
Recall@10:    0.0296
NDCG@10:      0.0468
```

(Previous 25-epoch model: P@10 0.0267, R@10 0.0242, NDCG 0.0386 — the extra
epochs + LR decay improved all three, but see the leakage warning below.)

### ⚠️ These numbers are optimistic — data leakage

The model is trained on **all 32 M ratings**, and the evaluation then holds out
20 % of each test user's ratings and scores against them. The model has already
seen those exact ratings, so the metrics measure "can the model reproduce what
it was trained on", not real ranking quality. The script prints this warning
every run.

**How to do it properly (backlog item):**

1. *Leave-one-out*: hold out one rated movie per user, retrain without it,
   rank and check whether it appears in the top-K. Expensive but standard.
2. *Time-based split*: train on ratings before timestamp T, evaluate after T —
   this also matches deployment reality (the dataset has timestamps, currently
   unused).
3. *k-fold cross-validation* over users — the honest way to compare
   hyperparameters.

Implementation notes: ratings are grouped once via
`groupby("user_idx").indices` (the previous version filtered the 32 M-row
DataFrame once per user ≈ 1000 full scans), and `K` is clamped to the number of
movies.

---

## 10. How to run

```powershell
# 0. Python 3.11+ (project currently uses 3.14), Node 20+
pip install -r requirements.txt

# 1. API key (optional but recommended)
copy .env.example .env      # then set TMDB_API_KEY=...

# 2. Train (≈ minutes-to-hours on 32 M ratings; only needed if you change the model)
python src/train.py         # writes model_weights.npz, movie_popularity.npy,
                            #         training_history.npz

# 3. Inspect training
python src/show_history.py           # print the run's RMSE/LR table
python src/plot_training.py          # saves + shows the RMSE curve
python src/evaluate_recommender.py
python src/tune_hyperparams.py       # resumable 5-config sweep (--reset to redo)

# 4. API  (http://127.0.0.1:5000)
python backend/app.py
#    FLASK_DEBUG=1 python backend/app.py   # enable the reloader/debugger

# 5. UI   (http://localhost:5173)
cd frontend
npm install
npm run dev
```

Environment variables:

| Variable | Default | Used by |
|---|---|---|
| `TMDB_API_KEY` | – | `backend/recommender.py` (from `.env`) |
| `FLASK_DEBUG` | `0` | `backend/app.py` |
| `FLASK_HOST` | `127.0.0.1` | `backend/app.py` |
| `FLASK_PORT` | `5000` | `backend/app.py` |
| `VITE_API_URL` | `http://127.0.0.1:5000` | frontend build/dev |

---

## 11. Fixed in this pass

| # | Problem | Fix |
|---|---|---|
| 1 | `requirements.txt` listed unused `streamlit`, was missing `flask`, `flask-cors`, `python-dotenv`, `requests`, `matplotlib` → fresh clone could not run | rewritten |
| 2 | Hardcoded `D:\…` absolute paths + CWD-relative `../` outputs | all paths derived from `Path(__file__).parent` |
| 3 | `POST /recommend` returned **HTTP 200** with an `error` key → frontend showed nothing | now `400`, and frontend surfaces `data.error` in all cases |
| 4 | Frontend had no timeout → infinite "Finding movies..." | 30 s `AbortController` + clear timeout message |
| 5 | Frontend API URL hardcoded | `VITE_API_URL` |
| 6 | `debug=True` always on; raw exceptions leaked to clients | `FLASK_DEBUG` env (default off) + generic 500 message |
| 7 | `validation_rmse` divided by `len(order)` (ZeroDivision); `fit` could restore `None` weights | guards + `validation_split` range check + "no validation set" path |
| 8 | `global_mean` computed over all ratings (leakage into validation) | computed from training rows only |
| 9 | Evaluation filtered the 32 M-row DataFrame per user | one-shot `groupby().indices` |
| 10 | Duplicate `normalize_title` + duplicated model-movie/popularity setup in `app.py` | single implementation imported from `recommender.py`; misalignment now raises instead of warning |
| 11 | `~84k`-row `iterrows()` genre loop and `np.percentile` recomputed on **every request** | precomputed genre masks + cached popularity threshold (≈ O(1) per request) |
| 12 | `lru_cache` permanently cached TMDB failures | TTL cache: stable results forever, failures 5 min |
| 13 | No input validation (unbounded movie list, unknown mood) | 50-title cap, mood/genre type checks |
| 14 | No random seed → unreproducible training | `np.random.seed(42)` |
| 15 | Numba kernels have no bounds checks → bad indices corrupt memory | `fit()` validates indices/lengths |
| 16 | `.gitignore` only had `.env` | added `__pycache__/`, `.vscode/`, `.kilo/`, logs, dist |
| 17 | `plot_training.py` could not regenerate the committed PNG | `savefig` added; paths relative to the file |

---

## 12. Training improvements (this pass)

| # | Change | Effect |
|---|---|---|
| 1 | 25 → 60 epochs with exponential LR decay (`0.95**epoch`, floor 1e-4) and patience 5 | val RMSE 0.7661 → **0.7635**; no overfitting (decay keeps epochs 24+ stable) |
| 2 | `fit(lr_decay=…, min_lr=…)` params + `lr_history` recorded per epoch | decay is inspectable in `training_history.npz` and printed per epoch |
| 3 | `train.py` hyperparameters hoisted to named constants (`N_FACTORS`, `EPOCHS`, `BASE_LR`, `LR_DECAY`, `MIN_LR`, `REG`, `VALIDATION_SPLIT`, `PATIENCE`) | one place to edit; the schedule is printed at startup |
| 4 | Atomic artifact writes (`tmp` + `os.replace`) for weights and history | interrupted training can't corrupt `model_weights.npz` |
| 5 | New `src/tune_hyperparams.py` — 5-config sweep, identical seed/split, results persisted to `tune_results.json` | resumable after a crash; confirmed the shipped config is the best of those tried (§5.1) |
| 6 | New `src/show_history.py` | print the last run's per-epoch RMSE/LR without opening a plot |

---

## 13. Known limitations & improvement backlog

**Machine learning**
1. ~~Under-trained: val RMSE still falling at epoch 25~~ **done** — 60 epochs +
   LR decay, now plateaued at 0.7635 (§5). Further gains need more factors
   with stronger λ, or a different model (ALS / kNN baseline).
2. Random rating split instead of a **time-based split** (timestamps are unused).
3. Evaluation is leaky (§9) — implement leave-one-out or temporal evaluation.
4. Hyperparameter search is a fixed 5-point grid over (k, λ, decay) — no
   lr/λ interaction, no k>40, no baseline comparison (item-item cosine, kNN,
   ALS).
5. Cold-start path ignores `b_i` and `μ` — a movie's learned bias could be a
   better popularity prior than raw rating counts.
6. Score components live on different scales (cosine ∈ [-1,1], genre ∈ [0,1],
   popularity ∈ [0,1]) with hand-picked weights — normalize per component or
   learn the weights.
7. Only 84,432 of 87,585 movies are recommendable (no ratings → no factors).
   Content-based features (genres/embeddings) could cover the rest.
8. Movies without a `(YYYY)` year are silently dropped by both year-based moods.

**Serving**
9. TMDB lookups are synchronous inside the request — a cold request can take
   seconds. Move to background prefetch of the top-N movies, or cache to disk.
10. Mood thresholds are hard-coded (`2015`, `1990`, p70) — dataset ends 2023.
11. No rate limiting, no auth, no request logging/metrics, no `/recommend`
    caching.
12. Single-process Flask with `debug` mode only — use gunicorn/Waitress (or
    FastAPI) for deployment.
13. Response can contain fewer than 10 items if a movie is missing from
    `movies.csv` (silently skipped).

**Frontend**
14. Single 726-line component — split into `GenrePicker`, `MoodPicker`,
    `MovieInput`, `ResultsGrid`, `MovieCard`.
15. No tests anywhere in the project (backend, frontend or ML).
16. No loading skeleton/empty state, no debounced autocomplete for movie titles
    (a `/search` endpoint exists only implicitly via title matching).
17. Default pre-selected genre `["Action"]` may surprise users.

**Project hygiene**
18. `src/*.py` are scripts executed on import — add `if __name__ == "__main__":`
    guards (importing `train` starts a full training run).
19. No CI, no Python linter/formatter (ruff/black), no type hints.
20. Repo root contains unrelated sibling projects; commit messages are
    non-descriptive.
21. `train.py` is not parameterised (no CLI/config file for epochs/λ/k).
22. `MatrixFactorization.predict/recommend` are unused by the backend, which
    reimplements scoring — either use them or delete them to avoid drift.

---

## 14. Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `ModuleNotFoundError: flask` | `pip install -r requirements.txt` |
| Posters are placeholders, trailers are YouTube **search** links | `TMDB_API_KEY` missing in `.env`, or TMDB unreachable (failures are only cached 5 min) |
| `WARNING: TMDB_API_KEY not found.` at startup | create `.env` from `.env.example` |
| Backend says `model_movies and Q are not aligned` | `movie_popularity.npy` / `model_weights.npz` come from different training runs — re-run `src/train.py` |
| Frontend: "Could not connect to the recommendation server" | start `python backend/app.py`, or set `VITE_API_URL` |
| Frontend: "The request timed out." | API too slow (TMDB) — check backend logs; 30 s abort |
| `ValueError: user_idx out of range for P …` | indices built for a different model — retrain or reload matching `movie_ids` |
| Crashes/NaN during training | almost always out-of-range indices hitting Numba's unchecked kernels — `fit()` now catches this |
| `git` pull asks for LFS | `git lfs install` (ratings.csv, tags.csv, model_weights.npz are LFS objects) |
