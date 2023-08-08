"""Convert the `full_kernel.csv` file to an adjacency matrix that can be read by Gephi."""
from pathlib import Path

import pandas as pd

HERE = Path(__file__).parent
FULL_KERNEL = HERE / "full_kernel.csv"
KERNEL_ADJ = HERE / "kernel-adj.csv"

full_df = pd.read_csv(FULL_KERNEL)
kernel_entries = full_df.loc[:, "K0":]
kernel_mat = kernel_entries.values
kernel_names = kernel_entries.columns

kdf = pd.DataFrame(kernel_mat, index=kernel_names, columns=kernel_names)
kdf.to_csv(KERNEL_ADJ)
