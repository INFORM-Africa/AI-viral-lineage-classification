import numpy as np

class ReliefF:
    def __init__(self, n_instances, n_neighbors=10):
        self.n_instances = n_instances
        self.n_neighbors = n_neighbors
        self.feature_importances_ = None
        self.selected_features_ = None
    
    def init_weights(self, n_features):
        self.feature_importances_ = np.zeros(n_features)

    def fit(self, X, y, n_select, is_discrete, n_iterations=10):
        self.init_weights(X.shape[1])
        subset_indices = np.random.choice(X.shape[0], self.n_instances, replace=False)
        X_subset = X[subset_indices]
        y_subset = y[subset_indices]
        classes = np.unique(y)
        classes_probs = np.array([np.mean(y == c) for c in classes])
        classes = np.unique(y_subset)

        for _ in range(n_iterations):
            R_index = np.random.choice(self.n_instances)
            R = X_subset[R_index]
            R_class = y_subset[R_index]

            updates = np.zeros(X.shape[1])
            for c in classes:
                class_instances = X_subset[y_subset == c]
                sort_indices = np.argsort(np.linalg.norm(class_instances - R, axis=1))
                class_nearest_neighbors = class_instances[sort_indices[:self.n_neighbors]]
                if c == R_class:
                    updates -= self.get_diffs(R, class_nearest_neighbors, is_discrete)
                else:
                    gamma = classes_probs[c] / (1 - classes_probs[R_class])
                    updates += gamma * self.get_diffs(R, class_nearest_neighbors, is_discrete)
            
            self.feature_importances_ += updates

        self.selected_features_ = np.argsort(self.feature_importances_)[-n_select:]

    def get_diffs(self, r, nearest_neighbors, is_discrete):
        denominator = self.n_neighbors * self.n_instances
        if is_discrete:
            return (r == nearest_neighbors).astype(int).sum(axis=0) / denominator
        
        denominator = denominator * (np.max(nearest_neighbors, axis=0) - np.min(nearest_neighbors, axis=0))
        return np.abs(r - nearest_neighbors).sum(axis=0)/ denominator

    def transform(self, X):
        return X[:, self.selected_features_]

    def fit_transform(self, X, y):
        self.fit(X, y)
        return self.transform(X)