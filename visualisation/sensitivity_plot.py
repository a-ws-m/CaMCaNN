"""Plot sensitivity metrics."""
from pathlib import Path
from typing import Dict, List

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

ROOT = Path(__file__).parents[1]

MODELS: Dict[str, str] = {
    # "ECFP All": "ecfp-all-{}-splits-trial-{}",
    # "ECFP Nonionics": "ecfp-nonionic-{}-splits-trial-{}",
    # "GNN All": "gnn-all-{}-splits-trial-{}",
    "GNN Nonionics": "gnn-nonionics-{}-splits-trial-{}",
}


def plot_sensitivity(dir_template: str, splits: List[int], repeats: List[int], plot_fname: str):
    """Make sensitivity plot for the model."""
    RMSE_NAME = "Test RMSE (log \si{\micro M})"
    rmse_values = []
    train_ratios = []
    for split, num_repeats in zip(splits, repeats):
        num_trials = split * num_repeats
        model_dirs = [ROOT / dir_template.format(split, trial) for trial in range(num_trials)]
        for model_dir in model_dirs:
            metrics_file = model_dir / "metrics.csv"
            predictions_file = model_dir / "predictions.csv"

            metrics = pd.read_csv(metrics_file, index_col=0, header=None)
            print(metrics)

            try:
                rmse_values.append(metrics.loc["root_mean_squared_error"].values.item())
            except KeyError:
                rmse_values.append(metrics.loc["test_rmse"].values.item())

            predictions = pd.read_csv(predictions_file, index_col=0)
            traintest_counts = predictions["traintest"].value_counts(normalize=True)
            train_ratios.append(traintest_counts["train"])

    rmse_df = pd.DataFrame({"Train ratio": train_ratios, RMSE_NAME: rmse_values})
    plt.figure()
    sns.regplot(rmse_df, x="Train ratio", y=RMSE_NAME, logx=True, x_bins=4)
    plt.savefig(plot_fname)


if __name__ == "__main__":
    for dir_template in MODELS.values():
        plot_sensitivity(
            dir_template,
            [2, 3, 4, 5],
            [3, 2, 1, 1],
            dir_template.split("-s")[0].format("sensitivity.pdf"),
        )
