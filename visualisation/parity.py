"""Utilities for plotting parities of predictions versus true log CMC values."""
from pathlib import Path
from typing import Tuple, Optional, NamedTuple, Literal, List

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm
from tqdm import tqdm

HERE = Path(__file__).parent


class ModelDataFrame(NamedTuple):
    df_path: str
    model_name: str
    task: Literal["Nonionics", "All"]
    pred_col: str = "pred"
    obs_col: str = "exp"
    has_uq: bool = False
    fname: Optional[str] = None


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


def joint_parity(
    model_dfs: List[ModelDataFrame],
    fname,
):
    """Plot the parity with error bars for a dataset."""
    PREDICTED_NAME = r"Predicted log CMC (\si{\micro M})"
    OBSERVED_NAME = r"Observed log CMC (\si{\micro M})"
    ALPHA = 0.6

    # MARKER_COLOUR = tuple(np.array([168, 190, 240]) / 255)

    plot_df = pd.DataFrame()
    for df_tuple in model_dfs:
        df = pd.read_csv(HERE / df_tuple.df_path)
        df["Model"] = df_tuple.model_name
        df["Task"] = df_tuple.task
        df["Subset"] = df["traintest"].str.capitalize()
        df[PREDICTED_NAME] = df[df_tuple.pred_col]
        df[OBSERVED_NAME] = df[df_tuple.obs_col]
        if df_tuple.has_uq:
            df["CI"] = df["stddev"] * 2
        plot_df = pd.concat([plot_df, df], ignore_index=True)

    with sns.axes_style("darkgrid"):
        g = sns.FacetGrid(
            data=plot_df, hue="Subset", row="Model", col="Task", margin_titles=True
        )
        g.map_dataframe(
            sns.scatterplot, OBSERVED_NAME, PREDICTED_NAME, alpha=ALPHA, markers=["o"] * 2
        )
        for (row_i, col_j, hue_k), data_ijk in g.facet_data():
            if data_ijk["CI"].notna().all():
                ax = g.facet_axis(row_i, col_j)

                plot_args = [
                    data_ijk[OBSERVED_NAME],
                    data_ijk[PREDICTED_NAME],
                    data_ijk["CI"],
                    None
                ]
                plot_kwargs = {
                    "marker": "",
                    "linestyle": "",
                    "alpha": ALPHA,
                    # "markers": ["o"] * 2,
                }
                plot_kwargs["color"] = g._facet_color(hue_k, None)
                for kw, val_list in g.hue_kws.items():
                    plot_kwargs[kw] = val_list[hue_k]

                # plot_kwargs["label"] = g.hue_names[hue_k]

                g._facet_plot(
                    plt.errorbar,
                    ax,
                    plot_args,
                    plot_kwargs,
                )

        g.add_legend()

        current_xlims = plt.xlim()
        current_ylims = plt.ylim()
        lim_pairs = list(zip(current_xlims, current_ylims))
        new_lims = [min(lim_pairs[0]), max(lim_pairs[1])]

        for ax in g.axes_dict.values():
            ax.plot(
                new_lims,
                new_lims,
                color="white",
                linestyle="--",
                marker="",
                zorder=0.5,
            )
            both_ticks = [ax.get_xticks(), ax.get_yticks()]
        g.set(xlim=new_lims, ylim=new_lims, aspect="equal")
        new_ticks = max(both_ticks, key=len)
        g.set(
            xticks=new_ticks,
            yticks=new_ticks,
            xlim=new_lims,
            ylim=new_lims,
            aspect="equal",
        )

        plt.savefig(fname, bbox_inches="tight")


def parity_plot(
    prob_df: pd.DataFrame,
    fname,
    pred_col: str = "pred",
    obs_col: str = "exp",
    stddev_col: Optional[str] = None,
    traintest: bool = False,
):
    """Plot the parity with error bars for a dataset."""
    PREDICTED_NAME = r"Predicted log CMC (\si{\micro M})"
    OBSERVED_NAME = r"Observed log CMC (\si{\micro M})"

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

    if traintest:
        hue = "Subset"
        map_df_kwargs = dict(markers=["o"] * 2)
    else:
        hue = None
        map_df_kwargs = dict()
    with sns.axes_style("darkgrid"):
        g = sns.FacetGrid(data=plot_df, hue=hue)
        if stddev_col is not None:
            g.map(
                plt.errorbar,
                OBSERVED_NAME,
                PREDICTED_NAME,
                "CI",
                marker="o",
                linestyle="",
                alpha=0.7,
                # markeredgewidth=1,
                # markeredgecolor="black",
                **map_df_kwargs
            )
        else:
            g.map_dataframe(
                sns.scatterplot,
                OBSERVED_NAME,
                PREDICTED_NAME,
                alpha=0.7,
                # linewidth=1,
                # edgecolor="black",
                **map_df_kwargs
            )

        if traintest:
            g.add_legend()

        current_xlims = plt.xlim()
        current_ylims = plt.ylim()
        lim_pairs = list(zip(current_xlims, current_ylims))
        new_lims = [min(lim_pairs[0]), max(lim_pairs[1])]

        plt.plot(
            new_lims, new_lims, color="white", linestyle="--", marker="", zorder=0.5
        )
        g.set(xlim=new_lims, ylim=new_lims, aspect="equal")
        both_ticks = [g.ax.get_xticks(), g.ax.get_yticks()]
        new_ticks = max(both_ticks, key=len)
        g.set(
            xticks=new_ticks,
            yticks=new_ticks,
            xlim=new_lims,
            ylim=new_lims,
            aspect="equal",
        )

        plt.savefig(fname, bbox_inches="tight")


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
    PER_FILE_PARAMS = {
        "gnn-all-nist-predictions.csv": dict(
            fname="gnn-all-nist-parity.pdf", obs_col="log CMC"
        ),
        "gnn-all-predictions.csv": dict(fname="gnn-all-parity.pdf", traintest=True),
        "gnn-nonionics-predictions.csv": dict(
            fname="gnn-nonionics-parity.pdf", traintest=True
        ),
        "ecfp-all-nist-predictions.csv": dict(
            fname="ecfp-all-nist-parity.pdf", obs_col="log CMC"
        ),
        "ecfp-all-predictions.csv": dict(fname="ecfp-all-parity.pdf", traintest=True),
        "ecfp-nonionics-predictions.csv": dict(
            fname="ecfp-nonionics-parity.pdf", traintest=True
        ),
    }
    # for path_, kwargs in tqdm(
    #     PER_FILE_PARAMS.items(), total=len(PER_FILE_PARAMS.keys())
    # ):
    #     df = pd.read_csv(HERE / path_)
    #     parity_plot(df, **kwargs)

    PER_FILE_TUPLES = [
        ModelDataFrame(
            df_path="gnn-all-predictions.csv",
            fname="gnn-all-parity.pdf",
            model_name="GNN",
            task="All",
        ),
        ModelDataFrame(
            df_path="gnn-nonionics-predictions.csv",
            fname="gnn-nonionics-parity.pdf",
            model_name="GNN",
            task="Nonionics",
        ),
        ModelDataFrame(
            df_path="ecfp-all-predictions.csv",
            fname="ecfp-all-parity.pdf",
            model_name="ECFP",
            task="All",
        ),
        ModelDataFrame(
            df_path="ecfp-nonionics-predictions.csv",
            fname="ecfp-nonionics-parity.pdf",
            model_name="ECFP",
            task="Nonionics",
        ),
        ModelDataFrame(
            df_path="gnn-uq-pred.csv",
            model_name="GNN/GP",
            task="All",
            has_uq=True,
        ),
        ModelDataFrame(
            df_path="gnn-uq-nonionics-pred.csv",
            model_name="GNN/GP",
            task="Nonionics",
            has_uq=True,
        ),
    ]
    joint_parity(PER_FILE_TUPLES, "joint-parity.pdf")
