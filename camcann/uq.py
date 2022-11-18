"""Uncertainty quantification with Gaussian Processes."""
import numpy as np

from typing import Dict, Tuple

import tensorflow as tf
from tensorflow.keras.models import Model
from scipy.stats import norm
from spektral.data import Loader
# from sklearn.gaussian_process import GaussianProcessRegressor
# from sklearn.gaussian_process.kernels import ConstantKernel, Matern, RationalQuadratic
from sklearn.preprocessing import StandardScaler

import gpflow as gpf

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
            latent_layer.set_weights(buffer.get_weights()[:len(latent_layer.get_weights())])

        self.latent_points = self.model.predict(
            train_data.load(), steps=train_data.steps_per_epoch
        ).astype(np.float64)
        self.latent_targets = np.array([graph.y for graph in train_data.dataset.graphs]).reshape(-1, 1)

        latent_dim = self.latent_points.shape[1]
        ls_start = np.array([1] * latent_dim)

        print(f"{latent_dim=}")
        if with_scaler:
            self.input_scaler = StandardScaler()
            self.latent_points = self.input_scaler.fit_transform(self.latent_points)
        else:
            self.input_scaler = None


        mlp_weights = loaded_model.layers[-1].get_weights()
        weights, bias = mlp_weights[-2:]
        weights = tf.cast(weights, tf.float64)
        bias = tf.cast(bias, tf.float64)
        self.mean_func = gpf.functions.Linear(weights, bias)

        self.kernel_func = gpf.kernels.RationalQuadratic(lengthscales=ls_start)
        self.gpr = gpf.models.GPR((self.latent_points, self.latent_targets), kernel=self.kernel_func, mean_function=self.mean_func)

        self.opt = gpf.optimizers.Scipy()
        self.opt.minimize(self.gpr.training_loss, self.gpr.trainable_variables)

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
        means, vars_ = self.gpr.predict_y(latent_points)
        stddevs = np.sqrt(vars_)
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
