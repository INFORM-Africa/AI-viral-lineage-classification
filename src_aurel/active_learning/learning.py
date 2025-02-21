from datetime import timedelta
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.exceptions import NotFittedError
from .tracked import get_tracked_lineages_df
import logging

class ActiveLearning(BaseEstimator, ClassifierMixin):

    def __init__(self, base_estimator, budget, decoder, lineages_df:pd.DataFrame):
        self.base_estimator = base_estimator
        self.budget = budget
        self.base_estimator_ = None
        self.query_strategy = None
        self.decoder = decoder
        self.tracked_lineages = get_tracked_lineages_df(lineages_df).lineage.values
        
        logging.basicConfig(format='%(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S', level=logging.DEBUG)

    def fit(self, X, y, dates, initial_date:str, timestep:int, query_strategy:str, report_frequency:int=10):
        assert len(X) == len(y), "X and y must have the same length"
        assert len(X) == len(dates), "X and dates must have the same length"
        assert timestep > 0, "timestep must be greater than 0"
        assert query_strategy in ['least_confident', 'margin_sampling', 'entropy'], "Invalid query strategy"
        
        self.query_strategy = query_strategy
        reported_lineages = {'lineages': [], 'date': []}
        reported_lineages_set = set()
        missed_lineages = {'lineages': [], 'date': []}
        missed_lineages_set = set()
        current_date = pd.to_datetime(initial_date)
        max_date = dates.max()
        detection_dates = {lineage: 'N/A' for lineage in self.tracked_lineages}
        timestep = timedelta(days=timestep)
        t = 0

        while current_date < max_date:
            if t%report_frequency == 0: logging.info(f"Current_date = {current_date}, t = {t}")
            end_date = current_date + timestep
            training_set_mask = dates <= current_date
            testing_set_mask = (dates > current_date) & (dates <= end_date)

            X_labeled, y_labeled = X[training_set_mask], y[training_set_mask]
            X_window, y_window = X[testing_set_mask], y[testing_set_mask]

            if X_window.shape[0] == 0:
                t += 1
                current_date = end_date
                continue

            if t%report_frequency == 0: logging.info(f"Training the base estimator on {len(X_labeled)} samples...")
            self.base_estimator_ = self.base_estimator.fit(X_labeled, y_labeled)

            if t%report_frequency == 0: logging.info(f"Predicting on {len(X_window)} samples...")
            most_uncertain_indices = self.find_most_uncertain_samples(X_window)
            most_uncertain_lineages = y_window[most_uncertain_indices]

            # Track reported lineages
            new_reported_lineages = list()
            for lineage in most_uncertain_lineages:
                decoded_lineage = self.decoder.inverse_transform([lineage])[0]
                if decoded_lineage in self.tracked_lineages and detection_dates[decoded_lineage] == 'N/A':
                        new_reported_lineages.append(decoded_lineage)
                        detection_dates[decoded_lineage] = current_date
            
            if t%report_frequency == 0: logging.info(f"Reporting lineages {set(new_reported_lineages)}")
            reported_lineages['lineages'].append(new_reported_lineages)
            reported_lineages['date'].append(current_date)
            reported_lineages_set.update(set(new_reported_lineages))

            # Identify missed lineages
            new_missed_lineages = list()
            for lineage in set(y_window):
                decoded_lineage = self.decoder.inverse_transform([lineage])[0]
                if decoded_lineage in self.tracked_lineages and detection_dates[decoded_lineage] == 'N/A':
                    if decoded_lineage not in reported_lineages_set:
                        new_missed_lineages.append(decoded_lineage)
                        

            if t%report_frequency == 0: logging.info(f"Missed lineages: {set(new_missed_lineages)}")
            missed_lineages['lineages'].append(new_missed_lineages)
            missed_lineages['date'].append(current_date)
            missed_lineages_set.update(set(new_missed_lineages))

            t += 1
            current_date = end_date

        lineages_detection_dates = {"lineage": [], "model_date": []}
        for lineage, model_date in detection_dates.items():
            lineages_detection_dates["lineage"].append(lineage)
            lineages_detection_dates["model_date"].append(model_date)

        return lineages_detection_dates, reported_lineages, missed_lineages

    def find_most_uncertain_samples(self, X_pool):
        if self.base_estimator_ is None:
            raise NotFittedError("This ActiveLearning instance is not fitted yet. Call 'fit' first.")
        
        if X_pool.shape[0] == 0:
            return []

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

    