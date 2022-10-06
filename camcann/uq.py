"""Uncertainty quantification with Gaussian Processes."""
import numpy as np

from typing import Dict, Tuple

from tensorflow.keras.models import Model
from scipy.stats import norm
from spektral.data import Loader
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern
from sklearn.preprocessing import StandardScaler

def nll(pred_mean: np.ndarray, pred_std: np.ndarray, true_vals: np.ndarray):
    """Get the negative log likelihood of a given set of predictions."""
    return -norm.logpdf(true_vals, loc=pred_mean, scale=pred_std).sum()


class GraphGPProcess:
    def __init__(
        self,
        graph_model: Model,
        train_data: Loader,
        loaded_model: Model,
        test_eval: bool = False,
        with_scaler: bool = True,
    ) -> None:
        """Create the latent space model and fit it."""
        self.model = graph_model
        self.model.predict(train_data.load(), steps=train_data.steps_per_epoch)
        loaded_model.predict(train_data.load(), steps=train_data.steps_per_epoch)
        for latent_layer, buffer in zip(self.model.layers, loaded_model.layers):
            latent_layer.set_weights(buffer.get_weights())

        self.latent_points = self.model.predict(
            train_data.load(), steps=train_data.steps_per_epoch
        )
        self.latent_targets = np.array([graph.y for graph in train_data.dataset.graphs])

        if with_scaler:
            self.input_scaler = StandardScaler()
            self.latent_points = self.input_scaler.fit_transform(self.latent_points)
        else:
            self.input_scaler = None

        self.kernel = ConstantKernel() * Matern(nu=1.5)

        self.gpr = GaussianProcessRegressor(
            self.kernel,
            normalize_y=True,
            n_restarts_optimizer=50,
            random_state=2022,
        ).fit(self.latent_points, self.latent_targets)

        self.log_likelihood = self.gpr.log_marginal_likelihood()

        if test_eval:
            evaluated_log_likelihood = self.evaluate(train_data)["log_likelihood"]
            print(
                f"Model's evaluate method ({evaluated_log_likelihood}) yields same value as sklearn's inbuilt ({self.log_likelihood}): "
                + str(np.allclose(self.log_likelihood, evaluated_log_likelihood))
            )

    def predict(self, test_data: Loader) -> Tuple[np.ndarray, np.ndarray]:
        """Predict means and standard deviations."""
        latent_points = self.model.predict(
            test_data.load(), steps=test_data.steps_per_epoch
        )
        if self.input_scaler is not None:
            latent_points = self.input_scaler.transform(latent_points)
        means, stddevs = self.gpr.predict(latent_points, return_std=True)
        return means, stddevs

    def evaluate(self, test_data: Loader) -> Dict[str, float]:
        """Evaluate performance on test data."""
        targets = np.array([graph.y for graph in test_data.dataset.graphs])
        means, stddevs = self.predict(test_data)

        neg_log_likelihood = nll(means, stddevs, targets)

        mae = np.mean(np.abs(targets - means))
        mse = np.mean(np.square(targets - means))
        rmse = np.sqrt(mse)

        return {
            "nll": neg_log_likelihood,
            "mae": mae,
            "mse": mse,
            "rmse": rmse,
        }
