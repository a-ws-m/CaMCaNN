from typing import List, Type, Optional

import keras_tuner
from spektral.layers import (
    GCNConv,
    GlobalAttentionPool,
    GlobalAttnSumPool,
    GlobalAvgPool,
    GlobalSumPool,
)
from spektral.layers.pooling.global_pool import GlobalPool
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.metrics import MeanAbsoluteError, RootMeanSquaredError
from tensorflow.keras.models import Model


class QinGNN(Model):
    """Graph neural network architecture described by Qin et al."""

    def __init__(
        self,
        channels: List[int] = [256] * 2,
        mlp_hidden_dim: List[int] = [256, 256],
        pool_func: Type[GlobalPool] = GlobalAvgPool,
        pooling_channels: Optional[int] = None,
        latent_model: bool = False,
    ):
        """Initialize model layers."""
        super().__init__()
        self.graph_layers: List[GCNConv] = [
            GCNConv(channel, activation="relu") for channel in channels
        ]
        self.pool = pool_func(pooling_channels) if pooling_channels else pool_func()
        self.output_mlp = self.make_mlp(
            pooling_channels if pooling_channels else channels[-1],
            mlp_hidden_dim,
            latent_model,
        )

    def make_mlp(
        self, input_size: int, mlp_hidden_dim: List[int], latent_model: bool = False
    ) -> Model:
        """Make MLP postprocessing layers."""
        if latent_model:
            # Hacky test for now
            return lambda x: x

        dense_layers = [Dense(dim, activation="relu") for dim in mlp_hidden_dim]
        dense_layers.append(Dense(1))

        mlp_input = Input((input_size))
        mlp_prop = dense_layers[0](mlp_input)

        if len(dense_layers) == 1:
            mlp_output = mlp_prop
        else:
            for layer in dense_layers[1:-1]:
                mlp_prop = layer(mlp_prop)

            mlp_output = dense_layers[-1](mlp_prop)

        return Model(mlp_input, mlp_output)

    def call(self, inputs, training=None, mask=None):
        try:
            x, a, _ = inputs
        except ValueError:
            x, a = inputs

        for layer in self.graph_layers:
            x = layer((x, a))

        out = self.pool(x)
        return self.output_mlp(out)


def build_gnn(hp: keras_tuner.HyperParameters, latent_model: bool = False) -> Model:
    """Build a GNN using keras tuner."""
    HIDDEN_DIM_CHOICES = dict(min_value=64, max_value=320, step=64)

    num_channels = hp.Int("graph_layers", min_value=2, max_value=3)
    num_mlp_layers = hp.Int("mlp_hidden_layers", min_value=1, max_value=2)

    pool_funcs = {
        "global_avg_pool": GlobalAvgPool,
        "global_sum_pool": GlobalSumPool,
        "global_attn_pool": GlobalAttentionPool,
        "global_attn_sum_pool": GlobalAttnSumPool,
    }
    pool_func_key = hp.Choice("pooling_func", list(pool_funcs.keys()))
    pool_func = pool_funcs[pool_func_key]

    pool_channels = (
        hp.Int("pool_channels", **HIDDEN_DIM_CHOICES)
        if pool_func_key == "global_attn_pool"
        else None
    )

    graph_channels = [
        hp.Int(f"graph_channels_{i}", **HIDDEN_DIM_CHOICES) for i in range(num_channels)
    ]
    mlp_hidden_dim = [
        hp.Int(f"mlp_hidden_dim_{i}", **HIDDEN_DIM_CHOICES)
        for i in range(num_mlp_layers)
    ]

    model = QinGNN(
        graph_channels,
        mlp_hidden_dim,
        pool_func,
        pool_channels,
        latent_model=latent_model,
    )
    model.compile(
        optimizer="adam",
        loss="mse",
        metrics=[RootMeanSquaredError(), MeanAbsoluteError()],
    )
    return model
