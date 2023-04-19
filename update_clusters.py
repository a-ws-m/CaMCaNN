"""Update the `cluster` column in the datasets."""
from pathlib import Path

import pandas as pd

HERE = Path(__file__).parent
CLUSTER_FILE = HERE / "models/ecfp-all/clusters.csv"
DATASET_DIR = HERE / "camcann/data/datasets"
QIN_ALL_FILE = DATASET_DIR / "qin_all_results.csv"
QIN_NONIONICS_FILE = DATASET_DIR / "qin_nonionic_results.csv"

cluster_df = pd.read_csv(CLUSTER_FILE, index_col=0)
qin_all_df = pd.read_csv(QIN_ALL_FILE, index_col=0)
qin_nonionics_df = pd.read_csv(QIN_NONIONICS_FILE, index_col=0)

for target_df, target_file in [(qin_all_df, QIN_ALL_FILE), (qin_nonionics_df, QIN_NONIONICS_FILE)]:
    target_df["cluster"] = cluster_df["cluster"]
    target_df.to_csv(target_file)
