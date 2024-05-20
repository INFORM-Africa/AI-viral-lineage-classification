from datetime import timedelta
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.exceptions import NotFittedError
import logging

class ActiveLearning(BaseEstimator, ClassifierMixin):

    def __init__(self, base_estimator, budget, query_strategy='least_confident'):
        self.base_estimator = base_estimator
        self.query_strategy = query_strategy
        self.budget = budget
        self.base_estimator_ = None

    def fit(self, X, y, dates, initial_date:str, timestep:int):
        assert len(X) == len(y), "X and y must have the same length"
        assert len(X) == len(dates), "X and dates must have the same length"
        assert timestep > 0, "timestep must be greater than 0"

        reported_lineages = {'lineage': [], 'date': []}

        # Convert the initial date to a datetime object
        current_date = pd.to_datetime(initial_date)
        max_date = dates.max()
        timestep = timedelta(days=timestep)
        t = 0

        while current_date < max_date:
            end_date = current_date + timestep
            training_set_mask = dates <= current_date
            testing_set_mask = (dates > current_date) & (dates <= end_date)

            X_labeled, y_labeled = X[training_set_mask], y[training_set_mask]
            X_window, y_window = X[testing_set_mask], y[testing_set_mask]

            self.base_estimator_ = self.base_estimator.fit(X_labeled, y_labeled)

            # Query the most uncertain instances from the samples in the window
            most_uncertain_indices = self.find_most_uncertain_samples(X_window)

            # Report the most uncertain instances
            for indice in most_uncertain_indices:
                reported_lineages['lineage'].append(y_window[indice])
                reported_lineages['date'].append(current_date)

            t += 1
            current_date = end_date

        return self

    def find_most_uncertain_samples(self, X_pool):
        if self.base_estimator_ is None:
            raise NotFittedError("This ActiveLearning instance is not fitted yet. Call 'fit' first.")

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

        return query_indices
    
    def _log_uncertain_samples(self, X_pool, uncertainties):
        sorted_indices = np.argsort(uncertainties)[::-1]
        n_queries = min(self.budget, len(X_pool))
        query_indices = sorted_indices[:n_queries]

        return query_indices

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
