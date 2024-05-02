import pandas as pd
import numpy as np
from imblearn.under_sampling import RandomUnderSampler
from src_aurel.classifiers.utils import is_child_of, is_terminal_node


class LocalPerParentNodeClassifier:
    def __init__(self):
        self.hierarchy = None
        self.models = {}

    def _find_hierarchy(self, lineages, root="root"):
        hierarchy = {root : set()}
        for lineage in lineages:
            levels = lineage.split(".")
            hierarchy[root].add(levels[0])
            key = levels[0]
            for level in levels[1:]:
                if key not in hierarchy:
                    hierarchy[key] = set()

                value = key + "." + level
                hierarchy[key].add(value)
                key = value

        return hierarchy
    
    def _sample_dataset_indexes(self, node:str, labels:np.array):
        assert node in self.hierarchy, f"Node {node} not found in hierarchy"
        local_dataset_indexes = np.argwhere([is_child_of(x, node) for x in labels])
        return local_dataset_indexes.squeeze()
    
    def _sample_dataset(self, node:str, dataset:pd.DataFrame, under_sample:bool=True):
        assert node in self.hierarchy, f"Node {node} not found in hierarchy"
        local_dataset = dataset[dataset['pangolin_lineage'].apply(lambda x: is_child_of(x, node))]
        X = local_dataset['sequence']
        y = local_dataset['pangolin_lineage']
        if under_sample:
            rus = RandomUnderSampler(random_state=0, replacement=False)
            X_resampled, y_resampled = rus.fit_resample(X, y)
        return X_resampled, y_resampled if under_sample else X, y
    
    def _init_models(self, model, **params):
        for parent in self.hierarchy:
            self.models[parent] = model(**params)

    def fit(self, dataset, model, **params):
        assert self.hierarchy is None, "Classifier already fitted"
        self.hierarchy = self._find_hierarchy(lineages=dataset.pangolin_lineage)
        self._init_models(model, **params)
        
        # for parent in self.hierarchy:
        #     X_train, y_train = self._sample_dataset(parent, dataset)
        #     self.models[parent].fit(X_train, y_train)

        for parent in self.hierarchy:
            indexes = self._sample_dataset_indexes(parent, dataset.pangolin_lineage.values)
            X_train = dataset.iloc[indexes]['sequence']
            y_train = dataset.iloc[indexes]['pangolin_lineage']
            self.models[parent].fit(X_train, y_train)

    def predict(self, X):
        predictions = []
        for x in X:
            current_node = "root"
            while not is_terminal_node(current_node, self.hierarchy):
                current_node = self.models[current_node].predict([x])
        
            predictions.append(current_node)
                
        return predictions
    
    # def predict_proba(self, X):
    #     return self.classifier.predict_proba(X)
    
    # def score(self, X, y):
    #     return self.classifier.score(X, y)
    