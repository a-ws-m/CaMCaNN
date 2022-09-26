from typing import List
from spektral.layers import GCNConv, GlobalAvgPool, LaPool, TAGConv
from spektral.layers import ops
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model


class QinGNN(Model):
    """Graph neural network architecture described by Qin et al."""

    def __init__(self) -> None:
        """Initialize layers."""
        super().__init__()
        self.gcn_1 = GCNConv(256, "relu")
        self.gcn_2 = GCNConv(256, "relu")
        self.avg_pool = GlobalAvgPool()
        self.readout_1 = Dense(256, activation="relu")
        self.readout_2 = Dense(256, activation="relu")
        self.out = Dense(1)

        self.graph_layers: List[GCNConv] = [self.gcn_1, self.gcn_2]
        self.readout_layers = [self.readout_1, self.readout_2, self.out]

    def call(self, inputs, training=None, mask=None):
        """Call the model."""
        x, a, _ = inputs
        for layer in self.graph_layers:
            x = layer((x, a))

        x = self.avg_pool(x)

        for layer in self.readout_layers:
            x = layer(x)

        return x


class CoarseGNN(Model):
    """Graph neural network architecture with Laplacian Pooling."""

    def __init__(self) -> None:
        """Initialize layers."""
        super().__init__()
        self.tag_1 = TAGConv(256, K=2, activation="relu")
        self.tag_2 = TAGConv(256, K=2, activation="relu")

        self.la_pool = LaPool()

        self.tag_3 = TAGConv(256, K=2, activation="relu")
        self.tag_4 = TAGConv(256, K=2, activation="relu")

        self.avg_pool = GlobalAvgPool()

        self.readout_1 = Dense(256, activation="relu")
        self.readout_2 = Dense(256, activation="relu")

        self.out = Dense(1)

        self.full_graph_layers = [self.tag_1, self.tag_2]
        self.pooled_graph_layers = [self.tag_3, self.tag_4]

        self.readout_layers = [self.readout_1, self.readout_2, self.out]

    def call(self, inputs, training=None, mask=None):
        """Call the model."""
        x, a, _ = inputs
        for layer in self.full_graph_layers:
            x = layer((x, a))

        x, a = self.la_pool((x, a))

        for layer in self.pooled_graph_layers:
            x = layer((x, a))

        x = self.avg_pool(x)

        for layer in self.readout_layers:
            x = layer(x)

        return x
