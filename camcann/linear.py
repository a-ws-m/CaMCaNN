"""Linear CMC prediction model with feature selection."""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import BayesianRidge, ElasticNetCV, RidgeCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .data.featurise.ecfp import SMILESHashes
from .uq import nll

def get_unnormed_contribs(
    coefs: np.ndarray, scaler: StandardScaler, selector: SelectFromModel
) -> np.ndarray:
    """Get the contributions of the unnormalised subgraphs."""
    scaled_ones = scaler.transform(np.ones((1, scaler.n_features_in_)))
    scaled_ones = selector.transform(scaled_ones)
    return coefs * scaled_ones


@dataclass
class LinearResults:
    """Hold the results of an ElasticNet regression model.

    Args:
        train_rmse: Root mean squared error on the training subset.
        train_r2: R^2 score on the training subset.
        final_alpha: The best alpha for the initial feature selection model.
        final_l1_ratio: The ratio of L1 to L2 found using CV for the initial
            feature selection model.
        alpha: The best alpha identified during training.
        coefs: The weights of each subgraph from training.
        intercept: The learned intercept of the model.
        test_rmse: The testing RMSE.
        test_r2: The testing R^2.
        num_non_negligible: The number of non-negligible coefficients (> 1e-5).
        train_nll: The final negative log likelihood during training.
        test_nll: The negative log likelihood on the test data.

    """

    train_rmse: float
    train_r2: float
    selection_alpha: float
    selection_l1_ratio: float
    final_alpha: float
    coefs: np.ndarray = field(repr=False)
    intercept: float
    test_rmse: float
    test_r2: float
    num_non_negligible: int
    train_nll: Optional[float] = None
    test_nll: Optional[float] = None

    def get_unnormed_contribs(
        self, scaler: StandardScaler, selector: SelectFromModel
    ) -> np.ndarray:
        """Get the contributions of the unnormalised subgraphs."""
        return get_unnormed_contribs(self.coefs, scaler, selector)


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

        self.scaler = StandardScaler()
        self.fs_encv = ElasticNetCV(l1_ratio=[0.1, 0.5, 0.7, 0.9, 0.95, 0.99, 1])
        self.final_ridge = RidgeCV(np.linspace(0.1, 10, 100))

    def remove_low_freq_subgraphs(self, threshold: Union[float, int] = 1) -> int:
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
            threshold = int(np.floor(threshold * self.train_fps.shape[0]))

        has_group = self.train_fps > 0
        include_group = has_group.sum(axis=0) > threshold

        self.smiles_hashes.hash_df["selected"] = list(include_group)
        self.smiles_hashes.hash_df["above_threshold_occurance"] = list(include_group)

        self.train_fps_filtered = self._apply_low_freq_filter(self.train_fps)
        self.test_fps_filtered = self._apply_low_freq_filter(self.test_fps)

        return (~include_group).sum()

    def _apply_low_freq_filter(self, fps: np.ndarray) -> np.ndarray:
        """Apply the low frequency subgraph filter to given fingerprints."""
        return fps[
            :, self.smiles_hashes.hash_df["above_threshold_occurance"].to_numpy()
        ]

    def elastic_feature_select(self) -> LinearResults:
        """Feature selection using Elastic Net CV regularisation.

        Returns:
            The results of the Elastic Net search.

        """
        self.selector = SelectFromModel(self.fs_encv, threshold=1e-5)
        self.model = make_pipeline(self.scaler, self.selector, self.final_ridge)
        self.model.fit(self.train_fps_filtered, self.train_targets)

        support = self.selector.get_support()
        end_model_selector = SelectFromModel(self.final_ridge, threshold=1e-5, prefit=True)
        end_support = end_model_selector.get_support()
        support[support] = end_support
        self.smiles_hashes.set_regularised_selection(support)

        num_non_negligible = support.sum()

        train_rmse, train_r2 = self.evaluate(
            self.train_fps, self.train_targets
        ).values()
        test_rmse, test_r2 = self.evaluate(self.test_fps, self.test_targets).values()

        return LinearResults(
            train_rmse,
            train_r2,
            self.selector.estimator_.alpha_,
            self.selector.estimator_.l1_ratio_,
            self.final_ridge.alpha_,
            self.final_ridge.coef_,
            self.final_ridge.intercept_,
            test_rmse,
            test_r2,
            num_non_negligible,
        )

    def evaluate(self, fps: np.ndarray, targets: np.ndarray) -> Dict[str, float]:
        """Get the RMSE and R^2 values for a set of fingerprints and target values."""
        predictions = self.predict(fps)
        flat_targets = targets
        rmse = mean_squared_error(flat_targets, predictions, squared=False)
        r2 = r2_score(flat_targets, predictions)
        return {"rmse": rmse, "r2": r2}

    def predict(self, fps: np.ndarray) -> np.ndarray:
        """Get an array of predictions."""
        try:
            return self.model.predict(self._apply_low_freq_filter(fps)).flatten()
        except AttributeError:
            raise ValueError("Must first fit model.")


class ProbECFPModel(LinearECFPModel):
    """A linear model based on ECFPs, but using Bayesian ridge regression."""

    def __init__(
        self,
        smiles_hashes: SMILESHashes,
        train_fps: np.ndarray,
        train_targets: np.ndarray,
        test_fps: np.ndarray,
        test_targets: np.ndarray,
    ) -> None:
        """Initialize smiles hash dataframe."""
        super().__init__(
            smiles_hashes, train_fps, train_targets, test_fps, test_targets
        )
        self.ridge = BayesianRidge(compute_score=True)

    def ridge_model_train_test(self) -> Dict[str, Any]:
        """Train and test the Bayesian ridge regression model."""
        self.model = make_pipeline(self.scaler, self.selector, self.ridge)
        self.model.fit(self.train_fps_filtered, self.train_targets)

        self.test_predictions, self.test_stds = self.model.predict(
            self.test_fps_filtered, {"return_std": True}
        )

        test_rmse = mean_squared_error(
            self.test_targets, self.test_predictions, squared=False
        )
        test_nll = nll(self.test_predictions, self.test_stds, self.test_targets)

        self.results = dict(
            best_rmse=-self.ridge.best_score_,
            alpha=self.ridge.alpha_,
            coefs=self.ridge.coef_,
            test_rmse=test_rmse,
            train_nll=-self.ridge.scores_[-1],
            test_nll=test_nll,
        )
        self.smiles_hashes.set_weights(
            self.results.coefs,
            self.results.get_unnormed_contribs(self.scaler, self.selector),
        )
        return self.results
