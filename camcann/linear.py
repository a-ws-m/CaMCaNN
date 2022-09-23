"""Linear CMC prediction model with feature selection."""
from typing import Dict, List, Union, NamedTuple
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNetCV, RidgeCV
import numpy as np

from .data.featurise.ecfp import SMILESHashes


class ECFPResults(NamedTuple):
    """Contains the results for a single fold's evaluation."""

    test_rmse: float

    @property
    def train_cv_rmse(self) -> float:
        """Get the best RMSE from the train set cross-validation."""
        # mses = self.pipeline[-1].cv_values_
        # return np.min(np.mean(mses, axis=1))
        return -self.pipeline[-1].best_score_

    @property
    def reduced_num_features(self) -> int:
        """Get the number of features after final selection."""
        return self.pipeline[1].get_support().sum()

    @property
    def best_alpha(self) -> float:
        """Get the best alpha from ridge regression."""
        return self.pipeline[-1].alpha_

    def group_weights(self, all_groups: List[str]) -> Dict[str, float]:
        """Get the weights associated with each group after final training."""
        weights = self.pipeline[-1].coef_
        initial_idxs = np.flatnonzero(self.initial_groups.values)

        final_groups = self.pipeline[1].get_support()
        final_idxs = initial_idxs[final_groups]

        return {all_groups[idx]: weight for idx, weight in zip(final_idxs, weights)}


class LinearECFPModel:
    """Get weights associated with the most important subgraphs to predict CMC."""

    def __init__(
        self,
        smiles_hashes: SMILESHashes,
        train_fps: np.ndarray,
        train_targets: np.ndarray,
        test_fps: np.ndarray,
        test_targets: np.ndarray,
    ) -> None:
        """Initialize smiles hash dataframe."""
        self.smiles_hashes = smiles_hashes
        self.train_fps = train_fps
        self.train_targets = train_targets
        self.test_fps = test_fps
        self.test_targets = test_targets

    def remove_low_freq_subgraphs(self, threshold: Union[float, int]) -> int:
        """Amend the smiles hashes to remove those that only occur once in the training data.

        Args:
            threshold: How many to remove. If a float, remove subgraphs that
                occur in fewer than this fraction of molecules in the training data.
                If an int, remove subgraphs that do not occur more than this
                many times.

        Returns:
            The number of subgraphs removed.

        """
        if isinstance(threshold, float):
            threshold = int(np.floor(threshold * self.train_fps.size[0]))

        has_group = self.train_fps > 0
        include_group = has_group.sum() > threshold

        self.smiles_hashes.hash_df["selected"] = include_group
        self.smiles_hashes.hash_df["above_threshold_occurance"] = include_group
        return (~include_group).sum()

    def train(self):
        """Train the model"""
        #TODO


def single_fold_routine(train_idxs, test_idxs, linear: bool = True) -> FoldResults:
    """Perform a single fold training routine."""
    # Get train/test data split
    train_df = features_df.iloc[train_idxs]
    test_df = features_df.iloc[test_idxs]

    train_targets = all_targets.iloc[train_idxs]
    test_targets = all_targets.iloc[test_idxs]

    # 1. Remove features that only occur once
    has_group = train_df > 0
    include_group = has_group.sum() > 1

    train_feats = train_df.iloc[:, include_group.values]
    test_feats = test_df.iloc[:, include_group.values]

    # 2. Train initial model and 3. feature selection
    elastic_net = ElasticNetCV(l1_ratio=[0.1, 0.5, 0.7, 0.9, 0.95, 0.99, 1])
    selector = SelectFromModel(elastic_net, threshold="mean")

    alphas = np.logspace(-10, 10, 21)
    out = RidgeCV(scoring="neg_root_mean_squared_error", alphas=alphas)

    pipe = make_pipeline(StandardScaler(), selector, out)
    pipe.fit(train_feats, train_targets)

    test_pred = pipe.predict(test_feats)
    test_rmse = np.sqrt(mean_squared_error(test_targets, test_pred))

    return FoldResults(include_group, pipe, test_rmse)
