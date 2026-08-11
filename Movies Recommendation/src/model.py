import numpy as np

class MatrixFactorization:
    def __init__(self, n_users, n_items, n_factors=20, lr=0.01, reg=0.02):
        self.n_factors = n_factors
        self.lr = lr
        self.reg = reg
        self.P = np.random.normal(0, 0.1, (n_users, n_factors))  # user factors
        self.Q = np.random.normal(0, 0.1, (n_items, n_factors))  # item factors
        self.b_u = np.zeros(n_users)
        self.b_i = np.zeros(n_items)
        self.global_mean = 0.0

    def fit(self, user_idx, item_idx, ratings, epochs=15, verbose=True):
        self.global_mean = ratings.mean()
        n = len(ratings)

        for epoch in range(epochs):
            perm = np.random.permutation(n)
            total_err = 0

            for i in perm:
                u, it, r = user_idx[i], item_idx[i], ratings[i]

                pred = (self.global_mean + self.b_u[u] + self.b_i[it]
                        + self.P[u].dot(self.Q[it]))
                err = r - pred
                total_err += err ** 2

                # SGD updates
                self.b_u[u] += self.lr * (err - self.reg * self.b_u[u])
                self.b_i[it] += self.lr * (err - self.reg * self.b_i[it])

                p_u = self.P[u].copy()
                self.P[u] += self.lr * (err * self.Q[it] - self.reg * self.P[u])
                self.Q[it] += self.lr * (err * p_u - self.reg * self.Q[it])

            if verbose:
                rmse = np.sqrt(total_err / n)
                print(f"Epoch {epoch+1}/{epochs} - RMSE: {rmse:.4f}")

    def predict(self, u, i):
        return self.global_mean + self.b_u[u] + self.b_i[i] + self.P[u].dot(self.Q[i])

    def recommend(self, u, n_items, top_n=10, exclude=None):
        scores = self.global_mean + self.b_u[u] + self.b_i + self.P[u].dot(self.Q.T)
        if exclude:
            scores[list(exclude)] = -np.inf
        top_idx = np.argsort(scores)[::-1][:top_n]
        return top_idx, scores[top_idx]