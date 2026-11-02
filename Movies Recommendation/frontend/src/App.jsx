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

  const toggleGenre = (genre) => {
    setSelectedGenres((current) =>
      current.includes(genre)
        ? current.filter((g) => g !== genre)
        : [...current, genre]
    );
  };

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
        setError("Enter at least one movie.");
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
        throw new Error("The server returned an invalid response.");
      }

      // The backend sends an `error` field for failed
      // requests AND for some 200 responses, so check both.
      if (!response.ok) {
        throw new Error(
          data.error || "Could not get recommendations."
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
        setError("Nothing matched those filters.");
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
        setError("The request timed out. Try again.");
      } else {
        setError(
          err.message ||
            "Could not connect to the server."
        );
      }

    } finally {
      clearTimeout(timeoutId);
      setLoading(false);
    }
  };

  const handlePosterError = (event) => {
    // Prevent infinite image-error loop
    event.currentTarget.onerror = null;
    event.currentTarget.style.display = "none";
  };

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

  return (
    <div className="app">

      <header className="site-header">
        <div className="site-name">CineMatch</div>
      </header>

      <main className="page">

        <h1>Tell me what you've watched</h1>

        <p className="lede">
          One movie per line, then pick a few genres and a mood.
        </p>

        <textarea
          value={movies}
          onChange={(e) => setMovies(e.target.value)}
          aria-label="Movies you have watched, one per line"
          placeholder="e.g. The Matrix, Aliens"
        />

        <div className="field">
          <span className="field-label">Genres</span>

          <div className="tag-list">
            {genres.map((genre) => (
              <button
                key={genre}
                type="button"
                className={
                  selectedGenres.includes(genre)
                    ? "tag active"
                    : "tag"
                }
                onClick={() => toggleGenre(genre)}
              >
                {genre}
              </button>
            ))}
          </div>
        </div>

        <div className="field">
          <span className="field-label">Mood</span>

          <div className="tag-list">
            {moods.map((item) => (
              <button
                key={item}
                type="button"
                className={mood === item ? "tag active" : "tag"}
                onClick={() => setMood(item)}
              >
                {item}
              </button>
            ))}
          </div>
        </div>

        {error && <p className="error">{error}</p>}

        <button
          className="submit"
          onClick={getRecommendations}
          disabled={loading}
        >
          {loading ? "Loading..." : "Get recommendations"}
        </button>

        {matchedMovies.length > 0 && (
          <p className="based-on">
            Based on: {matchedMovies.join(", ")}
          </p>
        )}

        {unmatchedMovies.length > 0 && (
          <p className="unmatched">
            Couldn't find: {unmatchedMovies.join(", ")}
          </p>
        )}

        {recommendations.length > 0 && (
          <section className="results" id="recommendations">
            <h2>Results</h2>

            <div className="grid">
              {recommendations.map((movie, index) => {
                const yearMatch = movie.title
                  ? movie.title.match(/\((\d{4})\)/)
                  : null;

                const title = yearMatch
                  ? movie.title.replace(/\s*\(\d{4}\)/, "")
                  : movie.title;

                return (
                  <article
                    className="card"
                    key={
                      movie.movieId ||
                      `${movie.title}-${index}`
                    }
                  >
                    <div
                      className="poster"
                      onClick={() => openTrailer(movie)}
                      role={movie.trailer ? "button" : undefined}
                      tabIndex={movie.trailer ? 0 : undefined}
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
                      <div className="poster-fallback">
                        <span>{movie.title}</span>
                      </div>

                      {movie.poster && (
                        <img
                          src={movie.poster}
                          alt={`${movie.title} poster`}
                          loading={
                            index < 4 ? "eager" : "lazy"
                          }
                          onError={handlePosterError}
                        />
                      )}
                    </div>

                    <div className="info">
                      <h3>
                        {index + 1}. {title}
                        {yearMatch && (
                          <span className="year">
                            {" "}
                            {yearMatch[1]}
                          </span>
                        )}
                      </h3>

                      <p className="genres-line">
                        {(movie.genres || "").replace(
                          /\|/g,
                          " / "
                        )}
                      </p>

                      <p className="score">
                        {Number(movie.score || 0).toFixed(2)}{" "}
                        <span>match</span>
                      </p>

                      {movie.reasons &&
                        movie.reasons.length > 0 && (
                          <ul className="reasons">
                            {movie.reasons.map(
                              (reason, reasonIndex) => (
                                <li
                                  key={`${reason}-${reasonIndex}`}
                                >
                                  {reason}
                                </li>
                              )
                            )}
                          </ul>
                        )}

                      {movie.trailer && (
                        <button
                          type="button"
                          className="trailer-link"
                          onClick={() => openTrailer(movie)}
                        >
                          Trailer
                        </button>
                      )}
                    </div>
                  </article>
                );
              })}
            </div>
          </section>
        )}

      </main>

      <footer className="site-footer">
        Model: matrix factorization
      </footer>

    </div>
  );
}

export default App;
