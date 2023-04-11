"""Plot sensitivity metrics."""
from pathlib import Path
from typing import Dict, List

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

ROOT = Path(__file__).parents[1]

MODELS: Dict[str, str] = {
    # "ECFP All": "ecfp-all-{}",
    "ECFP Nonionics": "ecfp-nonionics-{}",
    "GNN All": "gnn-all-{}",
    "GNN Nonionics": "gnn-nonionics-{}",
}


def plot_sensitivity(dir_template: str, ratio_range: List[float], plot_fname: str):
    """Make sensitivity plot for the model."""
    RMSE_NAME = "Test RMSE (log \si{\micro M})"
    rmse_values = []
    true_ratios = []
    for ratio in ratio_range:
        MODEL_DIR = ROOT / dir_template.format(int(ratio * 10))
        METRICS_FILE = MODEL_DIR / "metrics.csv"
        PREDICTIONS_FILE = MODEL_DIR / "predictions.csv"

        metrics = pd.read_csv(METRICS_FILE, index_col=0, header=None)
        print(metrics)

        try:
            rmse_values.append(metrics.loc["root_mean_squared_error"].values.item())
        except KeyError:
            rmse_values.append(metrics.loc["test_rmse"].values.item())

        predictions = pd.read_csv(PREDICTIONS_FILE, index_col=0)
        traintest_counts = predictions["traintest"].value_counts(normalize=True)
        true_ratios.append(traintest_counts["train"])

    rmse_df = pd.DataFrame({"Train ratio": true_ratios, RMSE_NAME: rmse_values})
    sns.relplot(rmse_df, x="Train ratio", y=RMSE_NAME)
    plt.savefig(plot_fname)


if __name__ == "__main__":
    for dir_template in MODELS.values():
        plot_sensitivity(
            dir_template,
            list(map(lambda x: x / 10, range(5, 9))),
            dir_template.format("sensitivity.pdf"),
        )
