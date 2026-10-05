import { useState } from "react";
import "./App.css";

const API_BASE =
  import.meta.env.VITE_API_URL ||
  "http://127.0.0.1:5000";

const REQUEST_TIMEOUT_MS = 30000;

function App() {
  const [selectedGenres, setSelectedGenres] = useState(["Action"]);
  const [mood, setMood] = useState("Something popular");
  const [movies, setMovies] = useState("");

  const [recommendations, setRecommendations] = useState([]);
  const [matchedMovies, setMatchedMovies] = useState([]);
  const [unmatchedMovies, setUnmatchedMovies] = useState([]);

  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const genres = [
    "Action",
    "Adventure",
    "Animation",
    "Comedy",
    "Crime",
    "Drama",
    "Fantasy",
    "Horror",
    "Romance",
    "Sci-Fi",
    "Thriller",
  ];

  const moods = [
    "Something popular",
    "Hidden gem / underrated",
    "Recent releases",
    "Classic / older films",
  ];

  // ============================================================
  // GENRE SELECTION
  // ============================================================

  const toggleGenre = (genre) => {
    setSelectedGenres((current) =>
      current.includes(genre)
        ? current.filter((g) => g !== genre)
        : [...current, genre]
    );
  };

  // ============================================================
  // GET RECOMMENDATIONS
  // ============================================================

  const getRecommendations = async () => {
    setLoading(true);
    setError("");
    setRecommendations([]);
    setMatchedMovies([]);
    setUnmatchedMovies([]);

    const controller = new AbortController();

    const timeoutId = setTimeout(
      () => controller.abort(),
      REQUEST_TIMEOUT_MS
    );

    try {
      const movieList = movies
        .split("\n")
        .map((movie) => movie.trim())
        .filter(Boolean);

      if (movieList.length === 0) {
        setError("Please enter at least one movie.");
        return;
      }

      const response = await fetch(
        `${API_BASE}/recommend`,
        {
          method: "POST",

          headers: {
            "Content-Type": "application/json",
          },

          body: JSON.stringify({
            movies: movieList,
            genres: selectedGenres,
            mood: mood,
          }),

          signal: controller.signal,
        }
      );

      let data;

      try {
        data = await response.json();
      } catch {
        throw new Error(
          "The recommendation server returned an invalid response."
        );
      }

      // The backend sends an `error` field for failed
      // requests AND for some 200 responses, so check both.
      if (!response.ok) {
        throw new Error(
          data.error || "Failed to get recommendations."
        );
      }

      if (data.error) {
        setError(data.error);
      }

      const results = data.recommendations || [];

      setRecommendations(results);

      setMatchedMovies(
        data.matched_movies || []
      );

      setUnmatchedMovies(
        data.unmatched_movies || []
      );

      if (results.length === 0 && !data.error) {
        setError(
          "No recommendations matched your filters. Try different movies, genres or mood."
        );
        return;
      }

      // Scroll to recommendations
      setTimeout(() => {
        document
          .getElementById("recommendations")
          ?.scrollIntoView({
            behavior: "smooth",
            block: "start",
          });
      }, 100);

    } catch (err) {
      console.error(err);

      if (err.name === "AbortError") {
        setError(
          "The request timed out. Please try again."
        );
      } else {
        setError(
          err.message ||
          "Could not connect to the recommendation server."
        );
      }

    } finally {
      clearTimeout(timeoutId);
      setLoading(false);
    }
  };

  // ============================================================
  // POSTER ERROR FALLBACK
  // ============================================================

  const handlePosterError = (event, movie) => {
    // Prevent infinite image-error loop
    event.currentTarget.onerror = null;

    const fallbackTitle = encodeURIComponent(
      movie.title || "Movie"
    );

    event.currentTarget.src =
      `https://placehold.co/500x750/11131a/ffffff?text=${fallbackTitle}`;
  };

  // ============================================================
  // OPEN TRAILER
  // ============================================================

  const openTrailer = (movie) => {
    if (!movie.trailer) {
      return;
    }

    window.open(
      movie.trailer,
      "_blank",
      "noopener,noreferrer"
    );
  };

  // ============================================================
  // RENDER
  // ============================================================

  return (
    <div className="app">

      {/* ======================================================
          NAVBAR
      ====================================================== */}

      <nav className="navbar">

        <div className="logo">
          🎬 CineMatch
        </div>

      </nav>


      {/* ======================================================
          HERO
      ====================================================== */}

      <section className="hero">

        <div className="hero-content">


          <h1>
            Find your next
            <span> favorite movie.</span>
          </h1>

          <p className="hero-description">
            Tell us what you enjoy and we'll discover
            movies that match your taste.
          </p>

          <button
            className="hero-button"
            onClick={() => {
              document
                .getElementById("discover")
                ?.scrollIntoView({
                  behavior: "smooth",
                });
            }}
          >
            Start Discovering ↓
          </button>

        </div>

      </section>


      {/* ======================================================
          DISCOVER
      ====================================================== */}

      <main
        className="discover"
        id="discover"
      >

        <div className="section-heading">

          <p>
            PERSONALIZE YOUR EXPERIENCE
          </p>

          <h2>
            What are you in the mood for?
          </h2>

          <span>
            Give us a few preferences and we'll do the rest.
          </span>

        </div>


        {/* ==================================================
            GENRES
        ================================================== */}

        <div className="preference-section">

          <h3>
            01. Pick your genres
          </h3>

          <div className="genre-list">

            {genres.map((genre) => (

              <button
                key={genre}
                type="button"
                className={
                  selectedGenres.includes(genre)
                    ? "genre active"
                    : "genre"
                }
                onClick={() => toggleGenre(genre)}
              >
                {genre}
              </button>

            ))}

          </div>

        </div>


        {/* ==================================================
            MOOD
        ================================================== */}

        <div className="preference-section">

          <h3>
            02. Choose your mood
          </h3>

          <div className="mood-grid">

            {moods.map((item) => (

              <button
                key={item}
                type="button"
                className={
                  mood === item
                    ? "mood-card active"
                    : "mood-card"
                }
                onClick={() => setMood(item)}
              >

                <div className="mood-icon">

                  {item === "Something popular" && "🔥"}

                  {item === "Hidden gem / underrated" && "💎"}

                  {item === "Recent releases" && "✨"}

                  {item === "Classic / older films" && "🎞️"}

                </div>

                <div>

                  <strong>
                    {item}
                  </strong>

                  <p>

                    {item === "Something popular" &&
                      "Popular with movie lovers"}

                    {item === "Hidden gem / underrated" &&
                      "Discover something different"}

                    {item === "Recent releases" &&
                      "Movies from recent years"}

                    {item === "Classic / older films" &&
                      "Timeless movies worth watching"}

                  </p>

                </div>

              </button>

            ))}

          </div>

        </div>


        {/* ==================================================
            MOVIES
        ================================================== */}

        <div className="preference-section">

          <h3>
            03. Movies you've enjoyed
          </h3>

          <p className="section-description">
            Enter movies you've recently enjoyed.
            One movie per line.
          </p>

          <textarea
            value={movies}
            onChange={(e) =>
              setMovies(e.target.value)
            }
            aria-label="Movies you have enjoyed, one per line"
            placeholder={
              "Iron Man\nAvengers\nInterstellar"
            }
          />

        </div>


        {/* ==================================================
            ERROR
        ================================================== */}

        {error && (

          <div className="error-message">
            {error}
          </div>

        )}


        {/* ==================================================
            GENERATE BUTTON
        ================================================== */}

        <div className="recommend-container">

          <button
            className="recommend-button"
            onClick={getRecommendations}
            disabled={loading}
          >

            {loading
              ? "Finding movies..."
              : "✨ Generate Recommendations"}

          </button>

        </div>


        {/* ==================================================
            MATCHED MOVIES
        ================================================== */}

        {matchedMovies.length > 0 && (

          <div className="matched-section">

            <p className="matched-label">
              Based on your movies
            </p>

            <div className="matched-movies">

              {matchedMovies.map(
                (movie, index) => (

                  <span
                    className="matched-movie"
                    key={`${movie}-${index}`}
                  >
                    {movie}
                  </span>

                )
              )}

            </div>

          </div>

        )}


        {/* ==================================================
            UNMATCHED MOVIES
        ================================================== */}

        {unmatchedMovies.length > 0 && (

          <div className="unmatched-message">

            Couldn't find:

            {" "}

            {unmatchedMovies.join(", ")}

          </div>

        )}


        {/* ==================================================
            RECOMMENDATIONS
        ================================================== */}

        {recommendations.length > 0 && (

          <section
            className="recommendations-section"
            id="recommendations"
          >

            <div className="recommendations-heading">

              <p>
                YOUR PERSONALIZED RESULTS
              </p>

              <h2>
                🎯 Recommended for you
              </h2>

              <span>
                Based on your taste and preferences
              </span>

            </div>


            <div className="movie-grid">

              {recommendations.map(
                (movie, index) => (

                  <article
                    className={
                      index === 0
                        ? "movie-card featured"
                        : "movie-card"
                    }
                    key={
                      movie.movieId ||
                      `${movie.title}-${index}`
                    }
                  >

                    {/* ======================================
                        POSTER
                    ====================================== */}

                    <div
                      className="poster-container"
                      onClick={() =>
                        openTrailer(movie)
                      }
                      role={
                        movie.trailer
                          ? "button"
                          : undefined
                      }
                      tabIndex={
                        movie.trailer
                          ? 0
                          : undefined
                      }
                      onKeyDown={(event) => {

                        if (
                          event.key === "Enter" ||
                          event.key === " "
                        ) {

                          event.preventDefault();

                          openTrailer(movie);

                        }

                      }}
                    >

                      <img
                        src={
                          movie.poster ||
                          `https://placehold.co/500x750/11131a/ffffff?text=${encodeURIComponent(
                            movie.title || "Movie"
                          )}`
                        }
                        alt={
                          `${movie.title} poster`
                        }
                        className="movie-poster"
                        loading={
                          index < 4
                            ? "eager"
                            : "lazy"
                        }
                        onError={(event) =>
                          handlePosterError(
                            event,
                            movie
                          )
                        }
                      />


                      {/* ==================================
                          PLAY OVERLAY
                      ================================== */}

                      {movie.trailer && (

                        <div className="poster-overlay">

                          <div className="play-button">
                            ▶
                          </div>

                          <span>
                            Watch Trailer
                          </span>

                        </div>

                      )}


                      {/* ==================================
                          RANK
                      ================================== */}

                      <div className="movie-rank">
                        #{index + 1}
                      </div>

                    </div>


                    {/* ======================================
                        MOVIE INFORMATION
                    ====================================== */}

                    <div className="movie-info">

                      <h3>
                        {movie.title}
                      </h3>


                      {/* GENRES */}

                      <p className="movie-genres">
                        {movie.genres}
                      </p>


                      {/* SCORE */}

                      <div className="movie-score">

                        <span>
                          ⭐
                        </span>

                        <strong>
                          {Number(
                            movie.score || 0
                          ).toFixed(3)}
                        </strong>

                      </div>


                      {/* DIVIDER */}

                      <div className="movie-divider" />


                      {/* REASONS */}

                      {movie.reasons &&
                        movie.reasons.length > 0 && (

                        <div className="movie-reasons">

                          {movie.reasons.map(
                            (reason, reasonIndex) => (

                              <p
                                key={
                                  `${reason}-${reasonIndex}`
                                }
                              >
                                🎯 {reason}
                              </p>

                            )
                          )}

                        </div>

                      )}


                      {/* TRAILER BUTTON */}

                      {movie.trailer && (

                        <button
                          type="button"
                          className="trailer-button"
                          onClick={() =>
                            openTrailer(movie)
                          }
                        >
                          ▶ Watch Trailer
                        </button>

                      )}

                    </div>

                  </article>

                )
              )}

            </div>

          </section>

        )}

      </main>


      {/* ======================================================
          FOOTER
      ====================================================== */}

      <footer>

        <div>
          🎬 CineMatch
        </div>

        <span>
          Model: Matrix Factorization
        </span>

      </footer>

    </div>
  );
}

export default App;