"""
This script originates from the HiClass library (https://github.com/scikit-learn-contrib/hiclass/) 
Miranda, F.M., Köehnecke, N. and Renard, B.Y. (2023), HiClass: a Python Library for Local Hierarchical
Classification Compatible with Scikit-learn', Journal of Machine Learning Research, 24(29), pp. 1–17. 
Available at: https://jmlr.org/papers/v24/21-1518.html.
"""

"""
Local classifier per node approach.

Numeric and string output labels are both handled.
"""

import hashlib
import pickle
from copy import deepcopy
from os.path import exists

import networkx as nx
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_array, check_is_fitted

from .constant_classifier import ConstantClassifier
from .abs_hierarchical_classifier import HierarchicalClassifier
from .training_policies import IMPLEMENTED_POLICIES as policies


class LocalClassifierPerNode(BaseEstimator, HierarchicalClassifier):
    """
    Assign local classifiers to each node of the graph, except the root node.

    A local classifier per node is a local hierarchical classifier that fits one local binary classifier
    for each node of the class hierarchy, except for the root node.

    Examples
        >>> from hiclass import LocalClassifierPerNode
        >>> y = [['1', '1.1'], ['2', '2.1']]
        >>> X = [[1, 2], [3, 4]]
        >>> lcpn = LocalClassifierPerNode()
        >>> lcpn.fit(X, y)
        >>> lcpn.predict(X)
        array([['1', '1.1'],
        ['2', '2.1']])
    """

    def __init__(
        self,
        local_classifier: BaseEstimator = None,
        binary_policy: str = "siblings",
        verbose: int = 0,
        edge_list: str = None,
        replace_classifiers: bool = True,
        n_jobs: int = 1,
        bert: bool = False,
        tmp_dir: str = None,
    ):
        """
        Initialize a local classifier per node.

        Parameters
            local_classifier (BaseEstimator, default=LogisticRegression) : The local_classifier used to create the collection of local classifiers. 
                Needs to have fit, predict andclone methods.
            binary_policy ({"exclusive", "less_exclusive", "exclusive_siblings", "inclusive", "less_inclusive", "siblings"}, str, default="siblings"):
                Specify the rule for defining positive and negative training examples, using one of the following options:
                
                - `exclusive`: Positive examples belong only to the class being considered. All classes are negative examples, except for the selected class;
                - `less_exclusive`: Positive examples belong only to the class being considered. All classes are negative examples, except for the selected class and its descendants;
                - `exclusive_siblings`: Positive examples belong only to the class being considered. All sibling classes are negative examples;
                - `inclusive`: Positive examples belong only to the class being considered and its descendants. All classes are negative examples, except for the selected class, its descendants and ancestors;
                - `less_inclusive`: Positive examples belong only to the class being considered and its descendants. All classes are negative examples, except for the selected class and its descendants;
                - `siblings`: Positive examples belong only to the class being considered and its descendants. All siblings and their descendant classes are negative examples.
                See :ref:`Training Policies` for more information about the different policies.
            verbose (int, default=0) : Controls the verbosity when fitting and predicting.
                See https://verboselogs.readthedocs.io/en/latest/readme.html#overview-of-logging-levels for more information.
            edge_list (str, default=None) : Path to write the hierarchy built.
            replace_classifiers (bool, default=True) : Turns on (True) the replacement of a local classifier with a constant classifier when trained on only a single unique class.
            n_jobs (int, default=1) : The number of jobs to run in parallel. Only :code:`fit` is parallelized.
                If :code:`Ray` is installed it is used, otherwise it defaults to :code:`Joblib`.
            bert (bool, default=False) : Wether to skip scikit-learn's checks and sample_weight passing for BERT.
            tmp_dir (str, default=None) : Temporary directory to persist local classifiers that are trained. If the job needs to be restarted,
                it will skip the pre-trained local classifier found in the temporary directory.
        """
        super().__init__(
            local_classifier=local_classifier,
            verbose=verbose,
            edge_list=edge_list,
            replace_classifiers=replace_classifiers,
            n_jobs=n_jobs,
            classifier_abbreviation="LCPN",
            bert=bert,
            tmp_dir=tmp_dir,
        )
        self.binary_policy = binary_policy

    def fit(self, X, y, sample_weight=None):
        """
        Fit a local classifier per node.

        Parameters
            X ({array-like, sparse matrix} of shape (n_samples, n_features)) : The training input samples. Internally, its dtype will be converted
                to ``dtype=np.float32``. If a sparse matrix is provided, it will be converted into a sparse ``csc_matrix``.
            y (array-like of shape (n_samples, n_levels)) : The target values, i.e., hierarchical class labels for classification.
            sample_weight (array-like of shape (n_samples,), default=None) : Array of weights that are assigned to individual samples.
                If not provided, then each sample is given unit weight.

        Returns
            self (object) : Fitted estimator.
        """
        # Execute common methods necessary before fitting
        super()._pre_fit(X, y, sample_weight)

        # Initialize policy
        self._initialize_binary_policy()

        # Fit local classifiers in DAG
        super().fit(X, y)

        # TODO: Store the classes seen during fit

        # TODO: Add function to allow user to change local classifier

        # TODO: Add parameter to receive hierarchy as parameter in constructor

        # Return the classifier
        return self

    def predict(self, X):
        """
        Predict classes for the given data.

        Hierarchical labels are returned.

        Parameters
            X ({array-like, sparse matrix} of shape (n_samples, n_features)) : The input samples. Internally, its dtype will be converted
                to ``dtype=np.float32``. If a sparse matrix is provided, it will be converted into a sparse ``csr_matrix``.
            return_uncertainty (bool, default=False) : If True, return the uncertainty of the predictions.
        Returns
            y (ndarray of shape (n_samples,) or (n_samples, n_outputs)) : The predicted classes.
            uncertainty (ndarray of shape (n_samples,)) : The estimated uncertainty of the predictions.
        """
        # Check if fit has been called
        check_is_fitted(self)

        # Input validation
        if not self.bert:
            X = check_array(X, accept_sparse="csr", allow_nd=True, ensure_2d=False)
        else:
            X = np.array(X)

        # Initialize array that holds predictions
        y = np.empty((X.shape[0], self.max_levels_), dtype=self.dtype_)

        # TODO: Add threshold to stop prediction halfway if need be

        bfs = nx.bfs_successors(self.hierarchy_, source=self.root_)

        self.logger_.info("Predicting")

        for predecessor, successors in bfs:
            if predecessor == self.root_:
                mask = [True] * X.shape[0]
                subset_x = X[mask]
            else:
                mask = np.isin(y, predecessor).any(axis=1)
                subset_x = X[mask]
            if subset_x.shape[0] > 0:
                probabilities = np.zeros((subset_x.shape[0], len(successors)))
                for i, successor in enumerate(successors):
                    successor_name = str(successor).split(self.separator_)[-1]
                    # self.logger_.info(f"Predicting for node '{successor_name}'")
                    classifier = self.hierarchy_.nodes[successor]["classifier"]
                    positive_index = np.where(classifier.classes_ == 1)[0]
                    probabilities[:, i] = classifier.predict_proba(subset_x)[
                        :, positive_index
                    ][:, 0]
                highest_probability = np.argmax(probabilities, axis=1)
                prediction = []
                for i in highest_probability:
                    prediction.append(successors[i])
                level = nx.shortest_path_length(self.hierarchy_, self.root_, predecessor)
                prediction = np.array(prediction)
                y[mask, level] = prediction

        y = self._convert_to_1d(y)

        self._remove_separator(y)

        return y


    def _initialize_binary_policy(self):
        if isinstance(self.binary_policy, str):
            self.logger_.info(f"Initializing {self.binary_policy} binary policy")
            try:
                self.binary_policy_ = policies[
                    self.binary_policy.lower()
                ](self.hierarchy_, self.X_, self.y_, self.sample_weight_)
            except KeyError:
                self.logger_.error(
                    f"Policy {self.binary_policy} not implemented. Available policies are:\n"
                    + f"{list(policies.keys())}"
                )
                raise KeyError(f"Policy {self.binary_policy} not implemented.")
        else:
            self.logger_.error("Binary policy is not a string")
            raise ValueError(
                f"Binary policy type must str, not {type(self.binary_policy)}."
            )

    def _initialize_local_classifiers(self):
        super()._initialize_local_classifiers()
        local_classifiers = {}
        for node in self.hierarchy_.nodes:
            # Skip only root node
            if node != self.root_:
                local_classifiers[node] = {
                    "classifier": deepcopy(self.local_classifier_)
                }
        nx.set_node_attributes(self.hierarchy_, local_classifiers)

    def _fit_digraph(self, local_mode: bool = False, use_joblib: bool = False):
        self.logger_.info("Fitting local classifiers")
        nodes = list(self.hierarchy_.nodes)
        # Remove root because it does not need to be fitted
        nodes.remove(self.root_)
        self._fit_node_classifier(nodes, local_mode, use_joblib)

    @staticmethod
    def _fit_classifier(self, node):
        classifier = self.hierarchy_.nodes[node]["classifier"]
        if self.tmp_dir:
            md5 = hashlib.md5(node.encode("utf-8")).hexdigest()
            filename = f"{self.tmp_dir}/{md5}.sav"
            if exists(filename):
                (_, classifier) = pickle.load(open(filename, "rb"))
                self.logger_.info(
                    f"Loaded trained model for local classifier {node.split(self.separator_)[-1]} from file {filename}"
                )
                return classifier
        if self.verbose > 0: self.logger_.info(f"Training local classifier {node}")
        X, y, sample_weight = self.binary_policy_.get_binary_examples(node)
        unique_y = np.unique(y)
        if len(unique_y) == 1 and self.replace_classifiers:
            classifier = ConstantClassifier()
        if not self.bert:
            try:
                classifier.fit(X, y, sample_weight)
            except TypeError:
                classifier.fit(X, y)
        else:
            classifier.fit(X, y)
        self._save_tmp(node, classifier)
        return classifier

    def _clean_up(self):
        super()._clean_up()
        del self.binary_policy_
