"""Convert the FDG json output into a CSV file for reading."""
from pathlib import Path

import json
import pandas as pd

HERE = Path(__file__).parent
FULL_KERNEL = HERE / "full_kernel.csv"
FDG_JSON = HERE / "fdg.json"
FDG_CSV = HERE / "fdg.csv"


def get_xy_data(json_file=FDG_JSON) -> pd.DataFrame:
    """Get positional data from a JSON file."""
    with json_file.open("r") as f:
        data = json.load(f)

    trim_data = [
        {"index": int(entry["label"][1:]), "x": entry["x"], "y": entry["y"]}
        for entry in data["nodes"]
    ]
    df = pd.DataFrame(trim_data)
    df.set_index("index", drop=True, inplace=True)
    df.index.name = None
    return df


kernel_df = pd.read_csv(FULL_KERNEL, index_col=0)
xy_data = get_xy_data()

new_df = kernel_df.join(xy_data)
new_df.to_csv(FDG_CSV)
