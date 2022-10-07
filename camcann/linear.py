"""Linear CMC prediction model with feature selection."""
from optparse import Option
from pathlib import Path
from tokenize import Name
from typing import Dict, List, Optional, Union, NamedTuple
from sklearn.pipeline import make_pipeline
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNetCV, RidgeCV, BayesianRidge
import numpy as np

from .data.featurise.ecfp import SMILESHashes
from .uq import nll


class RidgeResults(NamedTuple):
    """Hold the results of a ridge regression models.

    Args:
        best_rmse: The best root mean squared error during training.
        alpha: The best alpha identified during training.
        coefs: The weights of each subgraph from training.
        test_rmse: The testing RMSE.
        train_nll: The final negative log likelihood during training.
        test_nll: The negative log likelihood on the test data.

    """

    best_rmse: float
    alpha: float
    coefs: np.ndarray
    test_rmse: float
    train_nll: Optional[float] = None
    test_nll: Optional[float] = None

    def get_unnormed_contribs(self, scaler: StandardScaler, selector: SelectFromModel) -> np.ndarray:
        """Get the contributions of the unnormalised subgraphs."""
        scaled_ones = scaler.transform(np.ones((1, scaler.n_features_in_)))
        scaled_ones = selector.transform(scaled_ones)
        return self.coefs * scaled_ones

    def __repr__(self) -> str:
        rep = (
            f"Best train RMSE: {self.best_rmse}\n"
            f"Best alpha: {self.alpha}\n"
            f"Test RMSE: {self.test_rmse}"
        )

        if self.train_nll is not None:
            rep += f"\nFinal train NLL: {self.train_nll}"
        if self.test_nll is not None:
            rep += f"\nTest train NLL: {self.test_nll}"

        return rep


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
        self.encv = ElasticNetCV(l1_ratio=[0.1, 0.5, 0.7, 0.9, 0.95, 0.99, 1])
        self.ridge = RidgeCV(scoring="neg_root_mean_squared_error")

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
            threshold = int(np.floor(threshold * self.train_fps.size[0]))

        has_group = self.train_fps > 0
        include_group = has_group.sum(axis=0) > threshold

        self.smiles_hashes.hash_df["selected"] = list(include_group)
        self.smiles_hashes.hash_df["above_threshold_occurance"] = list(include_group)

        self.train_fps_filtered = self._apply_low_freq_filter(self.train_fps)
        self.test_fps_filtered = self._apply_low_freq_filter(self.test_fps)

        return (~include_group).sum()
    
    def _apply_low_freq_filter(self, fps: np.ndarray) -> np.ndarray:
        """Apply the low frequency subgraph filter to given fingerprints."""
        return fps[:, self.smiles_hashes.hash_df["above_threshold_occurance"].to_numpy()]

    def elastic_feature_select(self) -> int:
        """Feature selection using Elastic Net CV regularisation.

        Returns:
            The number of subgraphs returned.

        """
        selection_pipeline = make_pipeline(self.scaler, self.encv)
        selection_pipeline.fit(self.train_fps_filtered, self.train_targets)

        self.selector = SelectFromModel(self.encv, threshold="mean", prefit=True)

        support = self.selector.get_support()
        self.smiles_hashes.set_regularised_selection(support)
        return (~support).sum()

    def ridge_model_train_test(self) -> RidgeResults:
        """Train and test the ridge regression model."""
        self.model = make_pipeline(self.scaler, self.selector, self.ridge)
        self.model.fit(self.train_fps_filtered, self.train_targets)

        self.test_predictions = self.model.predict(self.test_fps_filtered)
        test_rmse = mean_squared_error(self.test_targets, self.test_predictions, squared=False)
        self.results = RidgeResults(
            best_rmse=-self.ridge.best_score_,
            alpha=self.ridge.alpha_,
            coefs=self.ridge.coef_,
            test_rmse=test_rmse,
        )
        self.smiles_hashes.set_weights(
            self.results.coefs, self.results.get_unnormed_contribs(self.scaler, self.selector)
        )
        return self.results

    def predict(self, fps: np.ndarray) -> np.ndarray:
        """Get an array of predictions."""
        try:
            return self.model.predict(self._apply_low_freq_filter(fps))
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
        super().__init__(smiles_hashes, train_fps, train_targets, test_fps, test_targets)
        self.ridge = BayesianRidge(compute_score=True)

    def ridge_model_train_test(self) -> RidgeResults:
        """Train and test the Bayesian ridge regression model."""
        self.model = make_pipeline(self.scaler, self.selector, self.ridge)
        self.model.fit(self.train_fps_filtered, self.train_targets)

        self.test_predictions, self.test_stds = self.model.predict(self.test_fps_filtered, {"return_std": True})

        test_rmse = mean_squared_error(self.test_targets, self.test_predictions, squared=False)
        test_nll = nll(self.test_predictions, self.test_stds, self.test_targets)

        self.results = RidgeResults(
            best_rmse=-self.ridge.best_score_,
            alpha=self.ridge.alpha_,
            coefs=self.ridge.coef_,
            test_rmse=test_rmse,
            train_nll=-self.ridge.scores_[-1],
            test_nll=test_nll,
        )
        self.smiles_hashes.set_weights(
            self.results.coefs, self.results.get_unnormed_contribs(self.scaler, self.selector)
        )
        return self.results
