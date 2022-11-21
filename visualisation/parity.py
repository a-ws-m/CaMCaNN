"""Utilities for plotting parities of predictions versus true log CMC values."""
from pathlib import Path
from typing import Tuple, Optional

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm
from tqdm import tqdm


def calc_pis(
    residuals: np.ndarray, stddevs: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate probability intervals for given set of residuals and standard deviations."""
    norm_resids = residuals / stddevs
    predicted_pi = np.linspace(0, 1, 100)
    bounds = norm.ppf(predicted_pi)
    observed_pi = np.array([np.count_nonzero(norm_resids <= bound) for bound in bounds])
    observed_pi = observed_pi / norm_resids.size
    return predicted_pi, observed_pi


def parity_plot(
    prob_df: pd.DataFrame,
    fname,
    pred_col: str = "pred",
    obs_col: str = "exp",
    stddev_col: Optional[str] = None,
    traintest: bool = False
):
    """Plot the parity with error bars for a dataset."""
    PREDICTED_NAME = r"Predicted log CMC ($\mu$M)"
    OBSERVED_NAME = r"Observed log CMC ($\mu$M)"

    # MARKER_COLOUR = tuple(np.array([168, 190, 240]) / 255)

    plot_data = {
        PREDICTED_NAME: prob_df[pred_col],
        OBSERVED_NAME: prob_df[obs_col],
    }
    if stddev_col is not None:
        plot_data["CI"] = (prob_df[stddev_col] * 2,)
    if traintest:
        plot_data["Subset"] = prob_df["traintest"].str.capitalize()

    plot_df = pd.DataFrame(plot_data)

    g = sns.FacetGrid(data=plot_df, height=7, hue=("Subset" if traintest else None))
    if stddev_col is not None:
        g.map_dataframe(
            plt.errorbar,
            OBSERVED_NAME,
            PREDICTED_NAME,
            "CI",
            marker="o",
            linestyle="",
            alpha=0.7,
            markeredgewidth=1,
            markeredgecolor="black",
        )
    else:
        g.map_dataframe(
            sns.scatterplot,
            OBSERVED_NAME,
            PREDICTED_NAME,
            alpha=0.7,
            linewidth=1,
            edgecolor="black",
        )

    current_xlims = plt.xlim()
    current_ylims = plt.ylim()
    lim_pairs = list(zip(current_xlims, current_ylims))
    new_lims = [min(lim_pairs[0]), max(lim_pairs[1])]

    g.map(
        sns.lineplot,
        x=new_lims,
        y=new_lims,
        label="Ideal",
        color="black",
        linestyle="--",
        marker="",
    )
    g.set(xlim=new_lims, ylim=new_lims, aspect="equal")
    if traintest:
        g.add_legend()
    plt.savefig(fname, transparent=True, bbox_inches="tight")


def plot_calibration(
    prob_df: pd.DataFrame,
    fname,
    pred_col: str = "pred",
    obs_col: str = "exp",
    stddev_col: str = "stddev",
):
    """Plot a calibration curve for a given dataset."""
    PREDICTED_NAME = "Predicted cumulative distribution"
    OBSERVED_NAME = "Observed cumulative distribution"

    LINE_COLOUR = sns.color_palette()[0]
    FILL_COLOUR = tuple(np.array(LINE_COLOUR) * 1.5)

    residuals = prob_df[obs_col] - prob_df[pred_col]

    predicted_pi, observed_pi = calc_pis(residuals, prob_df[stddev_col])
    data = pd.DataFrame({PREDICTED_NAME: predicted_pi, OBSERVED_NAME: observed_pi})
    ax = sns.lineplot(
        data=data, x=PREDICTED_NAME, y=OBSERVED_NAME, color=LINE_COLOUR, label="Actual"
    )
    sns.lineplot(
        ax=ax,
        x=(0, 1),
        y=(0, 1),
        color="black",
        linestyle="--",
        marker="",
        label="Ideal",
    )
    ax.set(xlim=(0, 1), ylim=(0, 1), aspect="equal")
    ax.fill_between(predicted_pi, predicted_pi, observed_pi, color=FILL_COLOUR)
    plt.savefig(fname, transparent=True, bbox_inches="tight")


if __name__ == "__main__":
    HERE = Path(__file__).parent
    PER_FILE_PARAMS = {
        "gnn-all-nist-predictions.csv": dict(
            fname="gnn-all-nist-parity.pdf", obs_col="log CMC"
        ),
        "gnn-all-predictions.csv": dict(fname="gnn-all-parity.pdf", traintest=True),
        "gnn-nonionics-predictions.csv": dict(fname="gnn-nonionics-parity.pdf", traintest=True),
        "ecfp-all-nist-predictions.csv": dict(
            fname="ecfp-all-nist-parity.pdf", obs_col="log CMC"
        ),
        "ecfp-all-predictions.csv": dict(fname="ecfp-all-parity.pdf", traintest=True),
        "ecfp-nonionics-predictions.csv": dict(fname="ecfp-nonionics-parity.pdf", traintest=True),
    }
    for path_, kwargs in tqdm(PER_FILE_PARAMS.items(), total=len(PER_FILE_PARAMS.keys())):
        df = pd.read_csv(HERE / path_)
        parity_plot(df, **kwargs)
