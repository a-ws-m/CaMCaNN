"""Visualisation utilities."""
import matplotlib as mpl
import seaborn as sns

from .parity import *

LATEX_PREAMBLE = "\n".join([
    r"\usepackage{siunitx}",
    r"\usepackage[sfdefault,lining]{FiraSans} %% option 'sfdefault' activates Fira Sans as the default text font",
    r"\usepackage[fakebold]{firamath-otf}",
    r"\renewcommand*\oldstylenums[1]{{\firaoldstyle #1}}",
    r"\sisetup{detect-weight,detect-mode}"
])

mpl.use("pgf")
sns.set_theme(
    context="paper",
    style="white",
    rc={
        "text.usetex": True,
        "pgf.texsystem": "xelatex",
        "pgf.preamble": LATEX_PREAMBLE,
        "pgf.rcfonts": False,
    },
)
