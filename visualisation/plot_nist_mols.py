import pandas as pd
import re
from pathlib import Path
from rdkit.Chem import MolFromSmiles
from rdkit.Chem.Draw import MolsToGridImage
from scipy.stats import norm

HERE = Path(__file__).parent
OUT_SVG = HERE / "nist-underpred.svg"

nist_pred_df = pd.read_csv(HERE / "uq_nist_pred.csv", index_col=0)

nist_pred_df["resid"] = nist_pred_df["log CMC"] - nist_pred_df["pred"]
# Normalise with respect to standard deviations
nist_pred_df["norm resid"] = nist_pred_df["resid"] / nist_pred_df["stddev"]
nist_pred_df["cdf"] = norm.cdf(nist_pred_df["norm resid"])
nist_pred_df["Above 95% CI"] = nist_pred_df["cdf"] > 0.95
nist_pred_df["Above 95% CI"].sum() / len(nist_pred_df["Above 95% CI"])

outliers = nist_pred_df[nist_pred_df["Above 95% CI"]]
outlie_smiles = outliers.SMILES
outlie_mols = list(map(MolFromSmiles, outlie_smiles))
svg: str = MolsToGridImage(
    outlie_mols,
    3,
    (600, 300),
    useSVG=True,
    minFontSize=30,
    drawMolsSameScale=False,
    padding=0.2,
)
svg = re.sub(r"<rect.*\n", "", svg)
OUT_SVG.write_text(svg)
