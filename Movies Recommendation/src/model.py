import numpy as np
from numba import njit


@njit
def sgd_epoch(
    user_idx,
    item_idx,
    ratings,
    order,
    P,
    Q,
    b_u,
    b_i,
    global_mean,
    lr,
    reg
):
    total_err = 0.0

    for j in range(len(order)):

        i = order[j]

        u = user_idx[i]
        it = item_idx[i]
        r = ratings[i]

        # Prediction
        dot = 0.0

        for f in range(P.shape[1]):
            dot += P[u, f] * Q[it, f]

        pred = (
            global_mean
            + b_u[u]
            + b_i[it]
            + dot
        )

        # Error
        err = r - pred

        total_err += err * err

        # Bias updates
        b_u[u] += lr * (
            err - reg * b_u[u]
        )

        b_i[it] += lr * (
            err - reg * b_i[it]
        )

        # Factor updates
        for f in range(P.shape[1]):

            pu = P[u, f]
            qi = Q[it, f]

            P[u, f] += lr * (
                err * qi
                - reg * pu
            )

            Q[it, f] += lr * (
                err * pu
                - reg * qi
            )

    return total_err


@njit
def validation_rmse(
    user_idx,
    item_idx,
    ratings,
    order,
    P,
    Q,
    b_u,
    b_i,
    global_mean
):
    total_err = 0.0

    if len(order) == 0:
        return np.nan

    for j in range(len(order)):

        i = order[j]

        u = user_idx[i]
        it = item_idx[i]
        r = ratings[i]

        dot = 0.0

        for f in range(P.shape[1]):
            dot += P[u, f] * Q[it, f]

        pred = (
            global_mean
            + b_u[u]
            + b_i[it]
            + dot
        )

        err = r - pred

        total_err += err * err

    return np.sqrt(
        total_err / len(order)
    )


class MatrixFactorization:

    def __init__(
        self,
        n_users,
        n_items,
        n_factors=20,
        lr=0.01,
        reg=0.02
    ):

        self.n_factors = n_factors
        self.lr = lr
        self.reg = reg

        # User latent factors
        self.P = np.random.normal(
            0,
            0.1,
            (n_users, n_factors)
        ).astype(np.float32)

        # Movie latent factors
        self.Q = np.random.normal(
            0,
            0.1,
            (n_items, n_factors)
        ).astype(np.float32)

        # User bias
        self.b_u = np.zeros(
            n_users,
            dtype=np.float32
        )

        # Movie bias
        self.b_i = np.zeros(
            n_items,
            dtype=np.float32
        )

        self.global_mean = 0.0

        # Training history
        self.train_rmse_history = []
        self.val_rmse_history = []
        self.lr_history = []


    def fit(
        self,
        user_idx,
        item_idx,
        ratings,
        epochs=15,
        validation_split=0.2,
        patience=3,
        verbose=True,
        lr_decay=1.0,
        min_lr=1e-4,
        val_data=None
    ):

        if not 0.0 < lr_decay <= 1.0:

            raise ValueError(
                "lr_decay must be in (0, 1]."
            )

        if min_lr <= 0:

            raise ValueError("min_lr must be positive.")

        user_idx = np.asarray(
            user_idx,
            dtype=np.int32
        )

        item_idx = np.asarray(
            item_idx,
            dtype=np.int32
        )

        ratings = np.asarray(
            ratings,
            dtype=np.float32
        )

        # ----------------------------------------------------
        # Input validation
        #
        # The Numba kernels run without bounds checking, so an
        # out-of-range index would silently corrupt memory
        # instead of raising. Validate up front.
        # ----------------------------------------------------

        if not (
            len(user_idx)
            == len(item_idx)
            == len(ratings)
        ):

            raise ValueError(
                "user_idx, item_idx and ratings "
                "must have the same length."
            )

        if len(user_idx) == 0:

            raise ValueError(
                "No ratings to train on."
            )

        if (
            user_idx.min() < 0
            or user_idx.max() >= self.P.shape[0]
        ):

            raise ValueError(
                "user_idx out of range for P with "
                f"shape {self.P.shape}."
            )

        if (
            item_idx.min() < 0
            or item_idx.max() >= self.Q.shape[0]
        ):

            raise ValueError(
                "item_idx out of range for Q with "
                f"shape {self.Q.shape}."
            )

        n = len(ratings)

        if val_data is not None:
            val_user_idx, val_item_idx, val_ratings, val_order = val_data
            train_order = np.arange(n, dtype=np.int32)
            self.global_mean = float(ratings.mean())
            print(f"Training ratings: {n:,}")
            print(f"Validation ratings: {len(val_ratings):,}")
            has_validation = len(val_ratings) > 0
        else:
            order = np.arange(n, dtype=np.int32)
            np.random.shuffle(order)
            split = int(n * (1 - validation_split))
            train_order = order[:split]
            val_order = order[split:]
            if not 0.0 <= validation_split < 1.0:
                raise ValueError("validation_split must be in [0, 1).")
            if len(train_order) == 0:
                raise ValueError("validation_split is too large: no training ratings left.")
            self.global_mean = float(ratings[train_order].mean())
            print(f"Training ratings: {len(train_order):,}")
            print(f"Validation ratings: {len(val_order):,}")
            has_validation = len(val_order) > 0
            val_user_idx = user_idx
            val_item_idx = item_idx
            val_ratings = ratings

        # --------------------------------------------
        # Training history
        # --------------------------------------------

        self.train_rmse_history = []
        self.val_rmse_history = []
        self.lr_history = []

        # --------------------------------------------
        # Best model
        # --------------------------------------------

        best_val_rmse = np.inf

        best_P = None
        best_Q = None
        best_b_u = None
        best_b_i = None

        best_epoch = 0

        patience_counter = 0

        # --------------------------------------------
        # Training
        # --------------------------------------------

        for epoch in range(epochs):

            # Exponential learning-rate decay
            epoch_lr = max(
                self.lr * (lr_decay ** epoch),
                min_lr
            )

            self.lr_history.append(epoch_lr)

            np.random.shuffle(
                train_order
            )

            total_err = sgd_epoch(
                user_idx,
                item_idx,
                ratings,
                train_order,
                self.P,
                self.Q,
                self.b_u,
                self.b_i,
                self.global_mean,
                epoch_lr,
                self.reg
            )

            train_rmse = np.sqrt(
                total_err
                / len(train_order)
            )

            # ----------------------------------------
            # Validation
            # ----------------------------------------

            val_rmse = validation_rmse(
                val_user_idx,
                val_item_idx,
                val_ratings,
                val_order,
                self.P,
                self.Q,
                self.b_u,
                self.b_i,
                self.global_mean
            )

            # Save history
            self.train_rmse_history.append(
                train_rmse
            )

            self.val_rmse_history.append(
                val_rmse
            )

            # ----------------------------------------
            # Print
            # ----------------------------------------

            if verbose:

                print(
                    f"Epoch {epoch + 1}/{epochs} "
                    f"- LR: {epoch_lr:.5f} "
                    f"- Train RMSE: {train_rmse:.4f} "
                    f"- Val RMSE: {val_rmse:.4f}"
                )

            # ----------------------------------------
            # Best model
            # ----------------------------------------

            if has_validation:

                if val_rmse < best_val_rmse:

                    best_val_rmse = val_rmse

                    best_P = self.P.copy()
                    best_Q = self.Q.copy()

                    best_b_u = self.b_u.copy()
                    best_b_i = self.b_i.copy()

                    best_epoch = epoch + 1

                    patience_counter = 0

                else:

                    patience_counter += 1

                # ------------------------------------
                # Early stopping
                # ------------------------------------

                if patience_counter >= patience:

                    print(
                        f"Early stopping at "
                        f"epoch {epoch + 1}"
                    )

                    break

        # --------------------------------------------
        # Restore best model
        # --------------------------------------------

        if best_P is None:

            if has_validation:

                raise RuntimeError(
                    "Training produced no valid "
                    "validation score (all NaN?)."
                )

            print(
                "No validation set: "
                "keeping final epoch weights."
            )

            return

        self.P = best_P
        self.Q = best_Q

        self.b_u = best_b_u
        self.b_i = best_b_i

        print(
            f"Best epoch: {best_epoch}"
        )

        print(
            f"Best validation RMSE: "
            f"{best_val_rmse:.4f}"
        )


    def predict(self, u, i):

        return (
            self.global_mean
            + self.b_u[u]
            + self.b_i[i]
            + self.P[u].dot(
                self.Q[i]
            )
        )


    def recommend(
        self,
        u,
        n_items,
        top_n=10,
        exclude=None
    ):

        scores = (
            self.global_mean
            + self.b_u[u]
            + self.b_i
            + self.P[u].dot(
                self.Q.T
            )
        )

        if exclude is not None:

            scores[
                list(exclude)
            ] = -np.inf

        top_idx = np.argsort(
            scores
        )[::-1][:top_n]

        return (
            top_idx,
            scores[top_idx]
        )