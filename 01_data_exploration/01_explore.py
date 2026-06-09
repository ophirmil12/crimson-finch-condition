"""
Step 01 – Data exploration.
Loads both primary files, prints structure summaries, and saves distribution plots.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

from config import (
    PCA_FILE, POISSON_IND_FILE, FIGURES, RESULTS,
    ENERGY_VARS, HAEM_VARS, ENERGY_COLORS, HAEM_COLORS,
    ENERGY_LABELS, HAEM_LABELS, COL_MALE, COL_FEMALE,
    YEAR_MAP, YEAR_COLS, YEAR_LABELS, setup_style, ensure_dirs
)

setup_style()
ensure_dirs()


# ── Load data ─────────────────────────────────────────────────────────────────

pca_df = pd.read_csv(PCA_FILE)
poi_df = pd.read_csv(POISSON_IND_FILE)

# Decode Year into a short label and numeric index
poi_df["YearLabel"] = poi_df["Year"].map({k: v[0] for k, v in YEAR_MAP.items()})
poi_df["YearIdx"]   = poi_df["Year"].map(
    {"1(2006-07)": 0, "2(2007-08)": 1, "3(2008-09)": 2, "4(2009-10)": 3}
)
poi_df["SexLabel"] = poi_df["Sex"].map({0: "Female", 1: "Male"})


# ── Console summary ───────────────────────────────────────────────────────────

print("=" * 60)
print("PCA file — condition indices (already mean-centred)")
print(f"  Shape : {pca_df.shape[0]} birds × {pca_df.shape[1]} variables")
print(pca_df.describe().round(3).to_string())

print("\n" + "=" * 60)
print("Poisson-independent file — individual breeding records")
print(f"  Shape : {poi_df.shape[0]} records × {poi_df.shape[1]} columns")
print("\nKey columns overview:")
key_cols = ["Year", "Sex", "Stage", "Agecat", "PC1", "PC2", "NumberIndependent"]
print(poi_df[key_cols].describe(include="all").to_string())

print("\nYear × Sex breakdown:")
print(poi_df.groupby(["YearLabel", "SexLabel"])["NumberIndependent"].describe().round(2))

print("\nMissing values:")
print(poi_df[key_cols].isnull().sum())


# ── Figure 1 – Condition index distributions ─────────────────────────────────

all_vars   = ENERGY_VARS + HAEM_VARS
all_cols   = ENERGY_COLORS + HAEM_COLORS
all_labels = ENERGY_LABELS + HAEM_LABELS

fig, axes = plt.subplots(1, 5, figsize=(15, 4))
fig.suptitle("Condition index distributions (centred values)", fontweight="bold")

for ax, var, col, lbl in zip(axes, all_vars, all_cols, all_labels):
    data = pca_df[var].dropna()
    ax.hist(data, bins=20, color=col, edgecolor="white", linewidth=0.5)
    ax.axvline(0, color="black", linestyle="--", linewidth=0.8, label="mean")
    ax.set_title(lbl, fontsize=10)
    ax.set_xlabel("Centred value")
    ax.set_ylabel("Count" if ax is axes[0] else "")

plt.tight_layout()
fig.savefig(FIGURES / "01a_condition_distributions.png", dpi=150, bbox_inches="tight")
plt.close()
print("\nSaved: 01a_condition_distributions.png")


# ── Figure 2 – Response variable distribution ────────────────────────────────

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
fig.suptitle("Reproductive output: number of independent young", fontweight="bold")

# Overall count distribution
ax = axes[0]
vals = poi_df["NumberIndependent"].value_counts().sort_index()
ax.bar(vals.index, vals.values, color=YEAR_COLS[1], edgecolor="white")
ax.set_xlabel("Number of independent young")
ax.set_ylabel("Number of records")
ax.set_title("Overall distribution")

# By year
ax = axes[1]
for i, (raw_year, (label, col)) in enumerate(YEAR_MAP.items()):
    subset = poi_df[poi_df["Year"] == raw_year]["NumberIndependent"]
    ax.hist(subset, bins=range(0, int(poi_df["NumberIndependent"].max()) + 2),
            alpha=0.6, color=col, label=label, edgecolor="white")
ax.set_xlabel("Number of independent young")
ax.set_ylabel("Count")
ax.set_title("By breeding season")
ax.legend(fontsize=9)

plt.tight_layout()
fig.savefig(FIGURES / "01b_response_distribution.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: 01b_response_distribution.png")


# ── Figure 3 – PC scores by year and sex ─────────────────────────────────────

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
fig.suptitle("Pre-computed PC scores from paper (PC1 = haematological, PC2 = energy)",
             fontweight="bold")

for ax, pc, title in zip(axes, ["PC1", "PC2"], ["PC1 (haematological)", "PC2 (energy reserves)"]):
    for raw_year, (label, col) in YEAR_MAP.items():
        subset = poi_df[poi_df["Year"] == raw_year][pc].dropna()
        ax.hist(subset, bins=15, alpha=0.55, color=col, label=label, edgecolor="white")
    ax.set_xlabel(f"{pc} score")
    ax.set_ylabel("Count")
    ax.set_title(title)

# Single shared legend placed to the right of the second panel
handles, labels = axes[1].get_legend_handles_labels()
fig.legend(handles, labels, loc="center right", fontsize=9,
           bbox_to_anchor=(1.02, 0.5), title="Season", title_fontsize=9,
           frameon=True)

plt.tight_layout()
fig.subplots_adjust(right=0.83)
fig.savefig(FIGURES / "01c_pc_scores_by_year.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: 01c_pc_scores_by_year.png")


# ── Save tidy summary CSV ─────────────────────────────────────────────────────

summary = poi_df.groupby("YearLabel")["NumberIndependent"].agg(
    n="count", mean="mean", median="median", std="std", max="max"
).reset_index()
summary.to_csv(RESULTS / "01_year_summary.csv", index=False)
print("Saved: 01_year_summary.csv")

print("\n[01_explore] Done.\n")
