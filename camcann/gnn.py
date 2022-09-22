from spektral.layers import GCNConv, GlobalAvgPool, LaPool
from spektral.utils.convolution import gcn_filter
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
        self.readout_1 = Dense(256)
        self.readout_2 = Dense(256)
        self.out = Dense(1)

        self.graph_layers = [self.gcn_1, self.gcn_2]
        self.readout_layers = [self.readout_1, self.readout_2, self.out]

    def call(self, inputs, training=None, mask=None):
        """Call the model."""
        x, a, _ = inputs
        for layer in self.graph_layers:
            x = layer(x, a)

        x = self.avg_pool(x)

        for layer in self.readout_layers:
            x = layer(x)

        return x


class CoarseGNN(Model):
    """Graph neural network architecture with Laplacian Pooling."""

    def __init__(self) -> None:
        """Initialize layers."""
        super().__init__()
        self.gcn_1 = GCNConv(256, "relu")
        self.gcn_2 = GCNConv(256, "relu")

        self.la_pool = LaPool()

        self.gcn_3 = GCNConv(256, "relu")
        self.gcn_4 = GCNConv(256, "relu")

        self.avg_pool = GlobalAvgPool()

        self.readout_1 = Dense(256)
        self.readout_2 = Dense(256)

        self.out = Dense(1)

        self.full_graph_layers = [self.gcn_1, self.gcn_2]
        self.pooled_graph_layers = [self.gcn_3, self.gcn_4]

        self.readout_layers = [self.readout_1, self.readout_2, self.out]

    def call(self, inputs, training=None, mask=None):
        """Call the model."""
        x, a, _ = inputs
        lap = gcn_filter(a)
        for layer in self.full_graph_layers:
            x = layer(x, lap)

        x, a = self.la_pool(x, a)
        lap = gcn_filter(a)

        for layer in self.pooled_graph_layers:
            x = layer(x, lap)

        x = self.avg_pool(x)

        for layer in self.readout_layers:
            x = layer(x)

        return x
