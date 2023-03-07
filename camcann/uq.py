"""Uncertainty quantification with Gaussian Processes."""
import numpy as np

from typing import Dict, Tuple, Optional, Union, List

import tensorflow as tf
from tensorflow.keras.models import Model
from scipy.stats import norm
from spektral.data import Loader

# from sklearn.gaussian_process import GaussianProcessRegressor
# from sklearn.gaussian_process.kernels import ConstantKernel, Matern, RationalQuadratic
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from tqdm import trange

import gpflow as gpf

from .data.io import QinGraphData, GraphData


def nll(pred_mean: np.ndarray, pred_std: np.ndarray, true_vals: np.ndarray):
    """Get the negative log likelihood of a given set of predictions."""
    return -norm.logpdf(true_vals, loc=pred_mean, scale=pred_std).sum()


class ScaledLinearMeanFunc(gpf.mean_functions.Linear):
    def __init__(
        self,
        scaler: StandardScaler,
        A: gpf.base.TensorType = None,
        b: gpf.base.TensorType = None,
    ) -> None:
        super().__init__(A, b)
        self.scaler = scaler

    def __call__(self, X: gpf.base.TensorType) -> tf.Tensor:
        X_orig = self.scaler.inverse_transform(X)
        return super().__call__(X_orig)

class MLPScalerMeanFunc(gpf.mean_functions.MeanFunction):

    def __init__(self, scaler: Optional[StandardScaler], mlp: tf.keras.Model, name=None):
        super().__init__(name)
        self.mlp = mlp
        self.scaler = scaler
    
    def __call__(self, X: gpf.base.TensorType) -> tf.Tensor:
        if self.scaler is not None:
            orig_X = self.scaler.inverse_transform(X)
        else:
            orig_X = X
        return tf.cast(self.mlp(X), tf.float64)

class GraphGPProcess:
    def __init__(
        self,
        graph_model: Model,
        graph_data: QinGraphData,
        loaded_model: Model,
        with_scaler: bool = True,
        lin_mean_func: bool = False,
    ) -> None:
        """Create the latent space model and fit it."""
        self.graph_data = graph_data
        self.with_scaler = with_scaler
        self.model = graph_model
        self.model.predict(
            graph_data.optim_loader_no_shuffle.load(),
            steps=graph_data.optim_loader_no_shuffle.steps_per_epoch,
        )

        loaded_model.predict(
            graph_data.optim_loader_no_shuffle.load(),
            steps=graph_data.optim_loader_no_shuffle.steps_per_epoch,
        ).flatten()

        for latent_layer, buffer in zip(self.model.layers, loaded_model.layers):
            try:
                latent_layer.set_weights(buffer.get_weights())
            except ValueError:
                for latent_sublayer, buffer_sublayer in zip(
                    latent_layer.layers, buffer.layers
                ):
                    latent_sublayer.set_weights(buffer_sublayer.get_weights())

        optim_latent_points, optim_targets = self._get_latent_data(
            graph_data.optim_loader_no_shuffle, graph_data.optim_dataset
        )

        mlp_weights = loaded_model.layers[-1].get_weights()
        mean_func_weights, mean_func_bias = mlp_weights[-2:]

        if with_scaler:
            self.input_scaler = StandardScaler()
            optim_latent_points = self.input_scaler.fit_transform(
                optim_latent_points
            ).astype(np.float64)
            if lin_mean_func:
                self.mean_func = ScaledLinearMeanFunc(
                    self.input_scaler, mean_func_weights, mean_func_bias
                )
        else:
            self.input_scaler = None
            if lin_mean_func:
                self.mean_func = gpf.mean_functions.Linear(
                    mean_func_weights, mean_func_bias
                )

        self.mean_func = MLPScalerMeanFunc(self.input_scaler, loaded_model.layers[-1])
        gpf.set_trainable(self.mean_func, False)

        self.optim_gpr = self._make_gp_model(optim_latent_points, optim_targets)
        gpf.utilities.print_summary(self.optim_gpr)
        self.final_gpr = self.train()

        self.log_likelihood = self.final_gpr.log_marginal_likelihood()

    def _get_latent_data(self, graph_loader: Loader, graph_data: GraphData) -> Tuple[np.ndarray, np.ndarray]:
        """Get latent points and targets for a given loader."""
        targets = np.array([graph.y for graph in graph_data.graphs]).reshape(-1, 1)
        latent_points = self.model.predict(
            graph_loader.load(), steps=graph_loader.steps_per_epoch
        ).astype(np.float64)

        return latent_points, targets

    def _make_gp_model(
        self, latent_points: np.ndarray, targets: np.ndarray
    ) -> gpf.models.GPR:
        """Make a Gaussian Process Regression Model."""
        latent_dim = latent_points.shape[1]
        ls_start = np.array([1] * latent_dim)

        kernel_func = gpf.kernels.Matern12(lengthscales=ls_start)

        gpr = gpf.models.GPR(
            (latent_points, targets), kernel=kernel_func, mean_function=self.mean_func, noise_variance=1e-6
        )
        gpf.set_trainable(gpr.likelihood, False)
        return gpr

    def train(self, num_epochs: int = 50000, patience: int = 1000) -> gpf.models.GPR:
        """Train a GP with early stopping."""
        EVAL_FREQUENCY: int = 100
        opt = tf.keras.optimizers.Adam()

        def get_trainable_vars() -> List[float]:
            return [var.value() for var in self.optim_gpr.trainable_variables]

        def get_all_train_model(vars_: List[float]) -> gpf.models.GPR:
            # train_latent_points, train_targets = self._get_latent_data(
            #     self.graph_data.train_loader_no_shuffle, self.graph_data.train_dataset
            # )
            # final_model = self._make_gp_model(train_latent_points, train_targets)
            for value, var in zip(self.optim_gpr.trainable_variables, vars_):
                value.assign(var)
            return self.optim_gpr

        @tf.function
        def step() -> tf.Tensor:
            opt.minimize(
                self.optim_gpr.training_loss, self.optim_gpr.trainable_variables
            )

        patience_counter: int = 0
        best_nll = np.inf
        best_params = get_trainable_vars()
        tbar = trange(num_epochs, desc="Training GP model")

        for i in tbar:
            step()
            if i % EVAL_FREQUENCY == 0:
                nll = self._evaluate_gpr(self.optim_gpr, self.graph_data.val_loader, just_nll=True)
                if nll < best_nll:
                    patience_counter = 0
                    best_nll = nll
                    best_params = get_trainable_vars()
                    tbar.set_postfix_str(f"Best NLL: {best_nll:.2f}")
                else:
                    patience_counter += EVAL_FREQUENCY
                    if patience_counter >= patience:
                        break

        return get_all_train_model(best_params)

    def _predict_gpr(self, gpr_model: gpf.models.GPR, test_data: Loader) -> Tuple[np.ndarray, np.ndarray]:
        """Predict means and standard deviations."""
        latent_points = self.model.predict(
            test_data.load(), steps=test_data.steps_per_epoch, verbose=0
        ).astype(np.float64)
        if self.input_scaler is not None:
            latent_points = self.input_scaler.transform(latent_points)

        means, vars_ = gpr_model.predict_y(latent_points)
        means = means.numpy()
        vars_ = vars_.numpy()
        stddevs = np.sqrt(vars_)
        return means.flatten(), stddevs.flatten()

    def predict(self, test_data: Loader) -> Tuple[np.ndarray, np.ndarray]:
        """Predict means and standard deviations."""
        return self._predict_gpr(self.final_gpr, test_data)

    def _evaluate_gpr(self, gpr_model: gpf.models.GPR, test_data: Loader, just_nll: bool = False) -> Union[float, Dict[str, float]]:
        """Evaluate performance on test data."""
        targets = np.array([graph.y for graph in test_data.dataset.graphs])
        means, stddevs = self._predict_gpr(gpr_model, test_data)

        neg_log_likelihood = nll(means, stddevs, targets)

        if just_nll:
            return neg_log_likelihood

        mae = np.mean(np.abs(targets - means))
        mse = np.mean(np.square(targets - means))
        rmse = np.sqrt(mse)

        return {
            "nll": neg_log_likelihood,
            "mae": mae,
            "mse": mse,
            "rmse": rmse,
        }

    def evaluate(
        self, test_data: Loader, just_nll: bool = False
    ) -> Union[float, Dict[str, float]]:
        """Evaluate performance on test data."""
        return self._evaluate_gpr(self.final_gpr, test_data, just_nll)
    
    def pairwise_matrix(self, test_data: Loader) -> np.ndarray:
        """Compute the pairwise kernel values for all of the data in the loader."""
        latent_points = self.model.predict(test_data.load(), steps=test_data.steps_per_epoch, verbose=0).astype(np.float64)
        if self.input_scaler is not None:
            latent_points = self.input_scaler.transform(latent_points)
        
        return self.final_gpr.kernel.K(latent_points).numpy()
