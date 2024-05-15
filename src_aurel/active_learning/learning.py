import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.exceptions import NotFittedError

class ActiveLearning(BaseEstimator, ClassifierMixin):

    def __init__(self, base_estimator, budget, query_strategy='least_confident'):
        self.base_estimator = base_estimator
        self.budget = budget
        self.query_strategy = query_strategy
        self.base_estimator_ = None

    def fit(self, X, y, initial_pool_size=None, max_iter=1000):
        if initial_pool_size is None:
            initial_pool_size = int(len(X) * 0.1)  # Default to 10% of the data

        # Randomly select the initial labeled pool
        indices = np.random.choice(len(X), size=initial_pool_size, replace=False)
        X_labeled, y_labeled = X[indices], y[indices]
        X_pool, y_pool = np.delete(X, indices, axis=0), np.delete(y, indices)
        t = 0

        while len(X_labeled) < self.budget and t < max_iter:
            self.base_estimator_ = self.base_estimator.fit(X_labeled, y_labeled)

            # Query the most uncertain instances from the pool
            X_query = self.query(X_pool)
            y_query = np.array([int(input(f"Enter the label for instance {i}: ")) for i in range(len(X_query))])

            # Add the queried instances to the labeled pool
            X_labeled = np.concatenate([X_labeled, X_query])
            y_labeled = np.concatenate([y_labeled, y_query])

            # Remove the queried instances from the pool
            query_indices = np.array([np.where((X_pool == x).all(axis=1))[0][0] for x in X_query])
            X_pool = np.delete(X_pool, query_indices, axis=0)
            y_pool = np.delete(y_pool, query_indices)

            t += 1

        return self

    def query(self, X_pool):
        if self.base_estimator_ is None:
            raise NotFittedError("This UncertaintySamplingActiveLearning instance is not fitted yet. Call 'fit' first.")

        if self.query_strategy == 'least_confident':
            uncertainties = self._least_confident(X_pool)
        elif self.query_strategy == 'margin_sampling':
            uncertainties = self._margin_sampling(X_pool)
        elif self.query_strategy == 'entropy':
            uncertainties = self._entropy(X_pool)
        else:
            raise ValueError(f"Invalid query strategy: {self.query_strategy}")

        sorted_indices = np.argsort(uncertainties)[::-1]
        n_queries = min(self.budget, len(X_pool))
        query_indices = sorted_indices[:n_queries]

        return X_pool[query_indices]

    def _least_confident(self, X):
        """
        Calculate the least confident probabilities for the instances in X.
        """
        probs = self.base_estimator_.predict_proba(X)
        least_confident = 1 - np.max(probs, axis=1)
        return least_confident

    def _margin_sampling(self, X):
        """
        Calculate the margin sampling scores for the instances in X.
        """
        probs = self.base_estimator_.predict_proba(X)
        sorted_probs = np.partition(-probs, 1, axis=1)[:, :2]
        margins = sorted_probs[:, 0] - sorted_probs[:, 1]
        return margins

    def _entropy(self, X):
        """
        Calculate the entropy scores for the instances in X.
        """
        probs = self.base_estimator_.predict_proba(X)
        entropies = -np.sum(probs * np.log(probs + 1e-10), axis=1)
        return entropies
