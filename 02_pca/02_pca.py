"""
Step 02 – PCA on condition indices.
Reproduces the paper's PCA (correlation matrix, eigenvalue > 1 rule).
Saves loadings, explained variance, scree plot, and biplot.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from config import (
    PCA_FILE, FIGURES, RESULTS,
    ENERGY_VARS, HAEM_VARS, ENERGY_COLORS, HAEM_COLORS,
    ENERGY_LABELS, HAEM_LABELS,
    COL_PC1, COL_PC2, setup_style, ensure_dirs
)

setup_style()
ensure_dirs()

# ── Load centred condition indices ────────────────────────────────────────────

df = pd.read_csv(PCA_FILE)
print(f"PCA dataset: {df.shape[0]} birds, {df.shape[1]} variables")

all_vars   = HAEM_VARS + ENERGY_VARS          # order: PCV, Hb, SMI, Muscle, Fat
all_labels = HAEM_LABELS + ENERGY_LABELS
all_colors = HAEM_COLORS + ENERGY_COLORS

X = df[all_vars].dropna()

# The data are already mean-centred; scale to unit variance (correlation-matrix PCA)
scaler = StandardScaler(with_mean=True, with_std=True)
X_scaled = scaler.fit_transform(X)

# ── Run PCA ───────────────────────────────────────────────────────────────────

pca = PCA()
scores = pca.fit_transform(X_scaled)

eigenvalues  = pca.explained_variance_          # eigenvalues of correlation matrix
exp_var_pct  = pca.explained_variance_ratio_ * 100
loadings     = pca.components_.T                # shape (n_vars, n_components)

n_pcs = int(np.sum(eigenvalues >= 1.0))         # paper rule: eigenvalue ≥ 1
print(f"\nEigenvalues: {np.round(eigenvalues, 3)}")
print(f"PCs retained (eigenvalue ≥ 1): {n_pcs}")
print(f"Variance explained by retained PCs: {exp_var_pct[:n_pcs].sum():.1f}%")

# ── Print loadings table ──────────────────────────────────────────────────────

loading_df = pd.DataFrame(
    loadings[:, :n_pcs],
    index=all_vars,
    columns=[f"PC{i+1}" for i in range(n_pcs)]
)
loading_df.index = all_labels
print("\nPC loadings (correlation-matrix PCA):")
print(loading_df.round(3).to_string())

# Interpret axes
pc1_top = loading_df["PC1"].abs().idxmax()
pc2_top = loading_df["PC2"].abs().idxmax()
print(f"\nPC1 dominated by: {pc1_top}")
print(f"PC2 dominated by: {pc2_top}")

# Save loadings
loading_df.round(4).to_csv(RESULTS / "02_pca_loadings.csv")
pd.DataFrame({
    "PC": [f"PC{i+1}" for i in range(len(eigenvalues))],
    "Eigenvalue": eigenvalues.round(4),
    "Pct_variance": exp_var_pct.round(2),
    "Cumulative_pct": np.cumsum(exp_var_pct).round(2),
}).to_csv(RESULTS / "02_pca_eigenvalues.csv", index=False)
print("\nSaved: 02_pca_loadings.csv, 02_pca_eigenvalues.csv")


# ── Figure 1 – Scree plot ─────────────────────────────────────────────────────

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
fig.suptitle("PCA on condition indices (4-year dataset)", fontweight="bold")

ax = axes[0]
pc_nums = range(1, len(eigenvalues) + 1)
bar_cols = [COL_PC1 if i < n_pcs else "#cccccc" for i in range(len(eigenvalues))]
ax.bar(pc_nums, eigenvalues, color=bar_cols, edgecolor="white")
ax.axhline(1.0, color="black", linestyle="--", linewidth=1, label="Eigenvalue = 1")
ax.set_xlabel("Principal component")
ax.set_ylabel("Eigenvalue")
ax.set_title("Scree plot")
ax.legend()
ax.set_xticks(list(pc_nums))

ax = axes[1]
ax.bar(pc_nums, exp_var_pct, color=bar_cols, edgecolor="white")
ax.plot(list(pc_nums), np.cumsum(exp_var_pct), "o-", color="black",
        linewidth=1.5, markersize=5, label="Cumulative %")
ax.set_xlabel("Principal component")
ax.set_ylabel("Variance explained (%)")
ax.set_title("Variance explained")
ax.legend()
ax.set_xticks(list(pc_nums))

plt.tight_layout()
fig.savefig(FIGURES / "02a_scree_plot.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: 02a_scree_plot.png")


# ── Figure 2 – Loading bar chart for retained PCs ────────────────────────────

fig, axes = plt.subplots(1, n_pcs, figsize=(6 * n_pcs, 4))
if n_pcs == 1:
    axes = [axes]

pc_palette = [COL_PC1, COL_PC2]
for pc_idx, ax in enumerate(axes):
    vals = loading_df.iloc[:, pc_idx].values
    colors = [c if v >= 0 else tuple(x * 0.7 for x in c)
              for c, v in zip(all_colors, vals)]
    bars = ax.barh(all_labels, vals, color=colors, edgecolor="white")
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Loading")
    ax.set_title(
        f"PC{pc_idx+1}  ({exp_var_pct[pc_idx]:.1f}% variance)\n"
        f"eigenvalue = {eigenvalues[pc_idx]:.2f}"
    )
    x_span = max(np.abs(vals)) * 1.2
    pad = x_span * 0.06
    for bar, val in zip(bars, vals):
        if val >= 0:
            # positive bar: label outside (to the right)
            ax.text(val + pad, bar.get_y() + bar.get_height() / 2,
                    f"{val:.2f}", va="center", ha="left", fontsize=9)
        else:
            # negative bar: label inside (to the right of bar end, inside bar)
            ax.text(val + pad, bar.get_y() + bar.get_height() / 2,
                    f"{val:.2f}", va="center", ha="left", fontsize=9,
                    color="white")

plt.tight_layout()
fig.subplots_adjust(left=0.22)
fig.savefig(FIGURES / "02b_pca_loadings.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: 02b_pca_loadings.png")


# ── Figure 3 – Biplot (PC1 vs PC2) ───────────────────────────────────────────

fig, ax = plt.subplots(figsize=(7, 6))

ax.scatter(scores[:, 0], scores[:, 1], alpha=0.35, s=20,
           color="#999999", zorder=1)

# Arrow scale
scale = 2.5
# Per-label nudges to avoid overlapping text when arrows cluster
_nudge = {
    "Scaled mass (SMI)": (-0.05, -0.35),
    "Muscle score":      (-0.40,  0.12),
    "Fat score":         ( 0.30,  0.12),
    "PCV (%)":           ( 0.30, -0.15),
    "Haemoglobin (÷10)": ( 0.30,  0.20),
}
for i, (lbl, col) in enumerate(zip(all_labels, all_colors)):
    dx = loadings[i, 0] * scale
    dy = loadings[i, 1] * scale
    ax.annotate("", xy=(dx, dy), xytext=(0, 0),
                arrowprops=dict(arrowstyle="->", color=col, lw=2))
    nx, ny = _nudge.get(lbl, (0.15 if dx >= 0 else -0.15,
                               0.15 if dy >= 0 else -0.15))
    ax.text(dx + nx, dy + ny, lbl, color=col,
            fontsize=9, ha="center", va="center", fontweight="bold")

ax.axhline(0, color="black", linewidth=0.5, linestyle="--")
ax.axvline(0, color="black", linewidth=0.5, linestyle="--")
ax.set_xlabel(f"PC1 ({exp_var_pct[0]:.1f}% variance) — haematological")
ax.set_ylabel(f"PC2 ({exp_var_pct[1]:.1f}% variance) — energy reserves")
ax.set_title("PCA biplot — condition indices")

plt.tight_layout()
fig.savefig(FIGURES / "02c_pca_biplot.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: 02c_pca_biplot.png")

print("\n[02_pca] Done.\n")
