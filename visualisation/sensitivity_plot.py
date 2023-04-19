"""Plot sensitivity metrics."""
from pathlib import Path
from typing import Dict, List

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("darkgrid")

ROOT = Path(__file__).parents[1]
MODEL_DIR = ROOT / "models"

MODELS: Dict[str, str] = {
    "ECFP Qin-All": "ecfp-all-{}-splits-trial-{}",
    "GNN Qin-All": "gnn-all-{}-splits-trial-{}",
    "ECFP Qin-Nonionics": "ecfp-nonionic-{}-splits-trial-{}",
    "GNN Qin-Nonionics": "gnn-nonionics-{}-splits-trial-{}",
}


def plot_sensitivity(dir_templates: List[str], model_names: List[str], splits: List[int], repeats: List[int], plot_fname: str):
    """Make sensitivity plot for the model."""
    RMSE_NAME = "Test RMSE (log \si{\micro M})"
    rmse_values = []
    train_ratios = []
    model_name_map = []
    for dir_template, model_name in zip(dir_templates, model_names):
        for split, num_repeats in zip(splits, repeats):
            num_trials = split * num_repeats
            model_dirs = [MODEL_DIR / dir_template.format(split, trial) for trial in range(num_trials)]
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
                model_name_map.append(model_name)

    rmse_df = pd.DataFrame({"Train ratio": train_ratios, RMSE_NAME: rmse_values, "Model name": model_name_map})
    fg = sns.FacetGrid(data=rmse_df, col="Model name", col_wrap=2)
    fg.map_dataframe(sns.regplot, x="Train ratio", y=RMSE_NAME, logx=True, x_bins=4)
    fg.set_titles(col_template="{col_name}")
    fg.tight_layout()
    fg.savefig(plot_fname)


if __name__ == "__main__":
    plot_sensitivity(
        MODELS.values(),
        MODELS.keys(),
        [2, 3, 4, 5],
        [3, 2, 1, 1],
        "paper/images/sensitivity-plots.pdf",
    )
