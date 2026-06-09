"""
Project-wide configuration: paths, color palette, and shared constants.
All scripts import from here so changes propagate everywhere.
"""

from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent
RAW_DATA  = ROOT / "raw_data"
FIGURES   = ROOT / "figures"
RESULTS   = ROOT / "results"

PCA_FILE         = RAW_DATA / "4+yrs+data+PCA+31May2013.csv"
POISSON_IND_FILE = RAW_DATA / "4+yrs+data+poisson+independent+31May2013.csv"

# ── Colour palette (tab20) ────────────────────────────────────────────────────
# tab20 gives 20 distinct colours in pairs (dark/light).
# We assign each colour a fixed semantic purpose so every script is consistent.
_tab20 = plt.get_cmap("tab20").colors  # tuple of 20 (R,G,B)

# Condition indices
COL_SMI    = _tab20[0]   # blue        – Scaled Mass Index (energy)
COL_MUSCLE = _tab20[2]   # orange      – Muscle score (energy)
COL_FAT    = _tab20[4]   # green       – Fat score (energy)
COL_PCV    = _tab20[6]   # red         – Packed Cell Volume (haematological)
COL_HB     = _tab20[8]   # purple      – Haemoglobin (haematological)
COL_TP     = _tab20[10]  # brown       – Total plasma protein (haematological)
COL_HL     = _tab20[12]  # pink        – H:L ratio (haematological)

# PC axes
COL_PC1    = _tab20[14]  # grey        – PC1 (haematological axis)
COL_PC2    = _tab20[16]  # olive/tan   – PC2 (energy-reserve axis)

# Sex
COL_MALE   = _tab20[1]   # light blue  – Male
COL_FEMALE = _tab20[3]   # light orange – Female

# Year (4 breeding seasons)
YEAR_COLS = [_tab20[5], _tab20[7], _tab20[9], _tab20[11]]
YEAR_LABELS = ["2006-07", "2007-08", "2008-09", "2009-10"]
YEAR_MAP = {
    "1(2006-07)": ("2006-07", YEAR_COLS[0]),
    "2(2007-08)": ("2007-08", YEAR_COLS[1]),
    "3(2008-09)": ("2008-09", YEAR_COLS[2]),
    "4(2009-10)": ("2009-10", YEAR_COLS[3]),
}

# Convenience: energy vs haematological variable groupings
ENERGY_VARS = ["SMIcntr", "Musclecntr", "Fatcntr"]
HAEM_VARS   = ["PCV_percent_cntr", "Hbdiv10cntr"]

ENERGY_COLORS = [COL_SMI, COL_MUSCLE, COL_FAT]
HAEM_COLORS   = [COL_PCV, COL_HB]

ENERGY_LABELS = ["Scaled mass (SMI)", "Muscle score", "Fat score"]
HAEM_LABELS   = ["PCV (%)", "Haemoglobin (÷10)"]

# ── Plot defaults ─────────────────────────────────────────────────────────────
FIGSIZE_SINGLE = (6, 4)
FIGSIZE_WIDE   = (10, 4)
FIGSIZE_GRID   = (12, 9)
DPI = 150

def setup_style():
    """Apply consistent matplotlib style."""
    plt.rcParams.update({
        "figure.dpi": DPI,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.size": 13,
        "axes.titlesize": 14,
        "axes.labelsize": 13,
    })

def ensure_dirs():
    FIGURES.mkdir(exist_ok=True)
    RESULTS.mkdir(exist_ok=True)
