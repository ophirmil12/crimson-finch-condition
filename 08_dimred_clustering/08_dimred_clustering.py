"""
Step 08 – Dimensionality reduction & clustering.
Embeds the 5-dimensional condition space (SMI, Muscle, Fat, PCV, Hb) via
PCA, t-SNE, and UMAP, then runs K-means clustering to discover natural
groupings.  Each projection is coloured by biological traits to surface
which aspects of variation each method captures.

Figures
-------
08a  Condition landscape — PCA / t-SNE / UMAP coloured by reproductive output
08b  Biological stratification — projections coloured by Sex and Stage
08c  K-means clustering — optimal k, cluster profiles, outcome by cluster
08d  Year dynamics — projections coloured by breeding season
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from scipy.stats import spearmanr
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.stats import kruskal
import umap

from config import (
    POISSON_IND_FILE, FIGURES, RESULTS,
    ENERGY_VARS, HAEM_VARS, ENERGY_COLORS, HAEM_COLORS,
    ENERGY_LABELS, HAEM_LABELS,
    COL_PC1, COL_PC2, COL_SMI, YEAR_COLS, YEAR_MAP,
    COL_MALE, COL_FEMALE,
    setup_style, ensure_dirs,
)

setup_style()
ensure_dirs()

# ── Data ──────────────────────────────────────────────────────────────────────

df = pd.read_csv(POISSON_IND_FILE)
df["YearLabel"] = df["Year"].map({k: v[0] for k, v in YEAR_MAP.items()})
df = df.dropna(subset=["NumberIndependent", "SMIcntr", "Musclecntr", "Fatcntr",
                        "PCV_percent_cntr", "Hbdiv10cntr"])

COND_VARS  = ENERGY_VARS + HAEM_VARS           # 5 condition indices
COND_COLS  = ENERGY_COLORS + HAEM_COLORS
COND_LBLS  = ENERGY_LABELS + HAEM_LABELS

X_raw  = df[COND_VARS].values
scaler = StandardScaler()
X      = scaler.fit_transform(X_raw)           # shape (n, 5)
n      = len(df)

# Trait arrays (for colouring)
outcome    = df["NumberIndependent"].values
sex_int    = df["Sex"].values                  # 0=F, 1=M
year_lbl   = df["YearLabel"].values

STAGE_ORDER = ["pre", "I", "N", "L", "B", "post"]
stage_num   = df["Stage"].map({s: i for i, s in enumerate(STAGE_ORDER)}).fillna(2).values
stage_lbl   = df["Stage"].values

# ── Compute embeddings ────────────────────────────────────────────────────────

print("Running PCA …")
pca    = PCA(n_components=2, random_state=42)
Z_pca  = pca.fit_transform(X)
pc_var = pca.explained_variance_ratio_ * 100

print("Running t-SNE …")
tsne   = TSNE(n_components=2, perplexity=30, learning_rate="auto",
              init="pca", random_state=42, n_iter=1000)
Z_tsne = tsne.fit_transform(X)

print("Running UMAP …")
reducer = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1,
                    random_state=42)
Z_umap  = reducer.fit_transform(X)

EMBEDS = [
    ("PCA",   Z_pca,  f"PC1 ({pc_var[0]:.1f}%)", f"PC2 ({pc_var[1]:.1f}%)"),
    ("t-SNE", Z_tsne, "t-SNE 1",                  "t-SNE 2"),
    ("UMAP",  Z_umap, "UMAP 1",                   "UMAP 2"),
]

print("Embeddings done.\n")


# ── K-means: choose k ─────────────────────────────────────────────────────────

K_RANGE  = range(2, 7)
sil_scores, inertias = [], []
km_labels = {}
for k in K_RANGE:
    km = KMeans(n_clusters=k, n_init=20, random_state=42)
    lbl = km.fit_predict(X)
    km_labels[k] = lbl
    sil_scores.append(silhouette_score(X, lbl))
    inertias.append(km.inertia_)
    print(f"  k={k}: silhouette={sil_scores[-1]:.3f}  inertia={inertias[-1]:.1f}")

best_k    = K_RANGE.start + int(np.argmax(sil_scores))
km_best   = km_labels[best_k]
print(f"\nBest k by silhouette: {best_k}\n")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 08a — Condition landscape coloured by reproductive output
# ─────────────────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle(
    "Condition space (PCA / t-SNE / UMAP) — coloured by reproductive output",
    fontweight="bold", fontsize=14,
)

vmin, vmax = 0, outcome.max()

for ax, (name, Z, xl, yl) in zip(axes, EMBEDS):
    sc = ax.scatter(Z[:, 0], Z[:, 1],
                    c=outcome, cmap="viridis", vmin=vmin, vmax=vmax,
                    s=38, alpha=0.75, edgecolors="none", zorder=2)
    ax.set_xlabel(xl, fontsize=11)
    ax.set_ylabel(yl, fontsize=11)
    ax.set_title(name, fontsize=13, fontweight="bold")
    # Annotate best Spearman r with reproductive outcome across both axes
    r1, p1 = spearmanr(Z[:, 0], outcome)
    r2, p2 = spearmanr(Z[:, 1], outcome)
    best_r, best_p = max([(r1, p1), (r2, p2)], key=lambda x: abs(x[0]))
    ax.text(0.03, 0.97,
            f"best axis: rs={best_r:+.2f}, p={best_p:.3f}",
            transform=ax.transAxes, fontsize=8, va="top", color="dimgray")

cbar = fig.colorbar(sc, ax=axes[-1], fraction=0.046, pad=0.04)
cbar.set_label("Independent young", fontsize=10)

plt.tight_layout()
fig.savefig(FIGURES / "08a_projections_outcome.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: 08a_projections_outcome.png")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 08b — Biological stratification: Sex (row 1) and Stage (row 2)
# ─────────────────────────────────────────────────────────────────────────────

STAGE_PALETTE = {
    "pre":  "#4C72B0",
    "I":    "#DD8452",
    "N":    "#55A868",
    "L":    "#C44E52",
    "B":    "#8172B3",
    "post": "#937860",
}

fig, axes = plt.subplots(2, 3, figsize=(16, 9))
fig.suptitle("Biological stratification in condition space",
             fontweight="bold", fontsize=14)

for col, (name, Z, xl, yl) in enumerate(EMBEDS):
    # ── Row 0: sex ──
    ax = axes[0, col]
    for val, label, col_ in [(0, "Female", COL_FEMALE), (1, "Male", COL_MALE)]:
        mask = sex_int == val
        ax.scatter(Z[mask, 0], Z[mask, 1],
                   color=col_, label=label,
                   s=32, alpha=0.65, edgecolors="none")
    ax.set_xlabel(xl, fontsize=10)
    ax.set_ylabel(yl, fontsize=10)
    ax.set_title(f"{name} — Sex", fontsize=12)
    if col == 0:
        ax.legend(fontsize=9, loc="best")

    # ── Row 1: breeding stage ──
    ax = axes[1, col]
    stages_present = [s for s in STAGE_ORDER if s in df["Stage"].unique()]
    for s in stages_present:
        mask = stage_lbl == s
        ax.scatter(Z[mask, 0], Z[mask, 1],
                   color=STAGE_PALETTE[s], label=s,
                   s=32, alpha=0.65, edgecolors="none")
    ax.set_xlabel(xl, fontsize=10)
    ax.set_ylabel(yl, fontsize=10)
    ax.set_title(f"{name} — Stage", fontsize=12)
    if col == 0:
        ax.legend(fontsize=8, loc="best", title="Stage", title_fontsize=8)

plt.tight_layout()
fig.savefig(FIGURES / "08b_projections_sex_stage.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: 08b_projections_sex_stage.png")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 08c — K-means clustering
#  Panel 1: Silhouette scores to justify k choice
#  Panel 2: UMAP with best-k clusters
#  Panel 3: Cluster condition profiles (centroid bar chart)
#  Panel 4: Reproductive output by cluster (violin + KW p-value)
# ─────────────────────────────────────────────────────────────────────────────

CLUSTER_PALETTE = [
    "#E41A1C", "#377EB8", "#4DAF4A", "#984EA3", "#FF7F00", "#A65628"
][:best_k]

# Centroids in original (unscaled) space for interpretability
centroids_scaled = np.zeros((best_k, len(COND_VARS)))
for k in range(best_k):
    centroids_scaled[k] = X[km_best == k].mean(axis=0)
# Back-transform to centred (original) units
centroids_orig = scaler.inverse_transform(centroids_scaled)

# Cluster sizes and mean outcomes
cluster_means = {k: outcome[km_best == k].mean() for k in range(best_k)}
cluster_ns    = {k: (km_best == k).sum()          for k in range(best_k)}

# Sort clusters by mean outcome so the story is clear
order = np.argsort([cluster_means[k] for k in range(best_k)])
cluster_labels_sorted = [f"C{i+1}" for i in range(best_k)]

# Kruskal-Wallis across clusters
kw_groups = [outcome[km_best == k] for k in range(best_k)]
kw_stat, kw_p = kruskal(*kw_groups)

fig, axes = plt.subplots(1, 4, figsize=(20, 5))
fig.suptitle(f"K-means clustering of condition space  (best k={best_k}, silhouette)",
             fontweight="bold", fontsize=13)

# Panel 1: Silhouette vs k
ax = axes[0]
ax.plot(list(K_RANGE), sil_scores, "o-", color=COL_SMI, linewidth=2, markersize=7)
ax.axvline(best_k, color="crimson", linestyle="--", linewidth=1.5,
           label=f"Best k={best_k}")
ax.set_xlabel("Number of clusters (k)")
ax.set_ylabel("Silhouette score")
ax.set_title("Silhouette analysis")
ax.set_xticks(list(K_RANGE))
ax.legend(fontsize=9)

# Panel 2: UMAP with cluster colours
ax = axes[1]
Z = Z_umap
for k in range(best_k):
    mask = km_best == k
    lbl = f"C{k+1}  (n={cluster_ns[k]}, mean={cluster_means[k]:.2f} ind.)"
    ax.scatter(Z[mask, 0], Z[mask, 1],
               color=CLUSTER_PALETTE[k], label=lbl,
               s=38, alpha=0.75, edgecolors="none")
ax.set_xlabel("UMAP 1", fontsize=10)
ax.set_ylabel("UMAP 2", fontsize=10)
ax.set_title(f"UMAP — k={best_k} clusters", fontsize=12)
ax.legend(fontsize=8, loc="best")

# Panel 3: Cluster condition profiles
ax = axes[2]
x_idx = np.arange(len(COND_VARS))
bar_w = 0.8 / best_k
for ki, k in enumerate(order):
    offsets = x_idx + ki * bar_w - (best_k - 1) * bar_w / 2
    ax.bar(offsets, centroids_orig[k], width=bar_w * 0.9,
           color=CLUSTER_PALETTE[k], alpha=0.85, edgecolor="white",
           label=f"C{k+1}")
ax.axhline(0, color="black", linewidth=0.7)
ax.set_xticks(x_idx)
ax.set_xticklabels(COND_LBLS, fontsize=8, rotation=25, ha="right")
ax.set_ylabel("Mean centred value")
ax.set_title("Cluster condition profiles\n(centroid in original units)")
ax.legend(fontsize=8)

# Panel 4: Reproductive output by cluster (violin)
ax = axes[3]
parts = ax.violinplot(
    [outcome[km_best == k] for k in order],
    positions=range(best_k),
    showmedians=True,
    showextrema=False,
)
for i, (body, k) in enumerate(zip(parts["bodies"], order)):
    body.set_facecolor(CLUSTER_PALETTE[k])
    body.set_alpha(0.75)
parts["cmedians"].set_colors("white")
parts["cmedians"].set_linewidth(2)
ax.set_xticks(range(best_k))
ax.set_xticklabels([f"C{k+1}" for k in order], fontsize=10)
ax.set_ylabel("Independent young")
ax.set_title(f"Outcome by cluster\n(KW p = {kw_p:.4f})")
# overlay individual points
for i, k in enumerate(order):
    jit = np.random.default_rng(i).uniform(-0.08, 0.08, (km_best == k).sum())
    ax.scatter(np.full((km_best == k).sum(), i) + jit,
               outcome[km_best == k],
               color=CLUSTER_PALETTE[k], alpha=0.3, s=14, edgecolors="none")

plt.tight_layout()
fig.savefig(FIGURES / "08c_clustering.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: 08c_clustering.png")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 08d — Year dynamics: projections coloured by breeding season
# ─────────────────────────────────────────────────────────────────────────────

YEAR_ORDER = ["2006-07", "2007-08", "2008-09", "2009-10"]

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("Temporal dynamics — condition space coloured by breeding season",
             fontweight="bold", fontsize=14)

for ax, (name, Z, xl, yl) in zip(axes, EMBEDS):
    for yi, yr in enumerate(YEAR_ORDER):
        mask = year_lbl == yr
        ax.scatter(Z[mask, 0], Z[mask, 1],
                   color=YEAR_COLS[yi], label=yr,
                   s=36, alpha=0.70, edgecolors="none")
        # centroid marker
        ax.scatter(Z[mask, 0].mean(), Z[mask, 1].mean(),
                   color=YEAR_COLS[yi], s=160, marker="D",
                   edgecolors="white", linewidths=1.5, zorder=5)
    ax.set_xlabel(xl, fontsize=10)
    ax.set_ylabel(yl, fontsize=10)
    ax.set_title(f"{name} — Season", fontsize=12)
    if ax is axes[0]:
        handles = [mpatches.Patch(color=YEAR_COLS[i], label=YEAR_ORDER[i])
                   for i in range(len(YEAR_ORDER))]
        ax.legend(handles=handles, fontsize=8, loc="best", title="Season")

plt.tight_layout()
fig.savefig(FIGURES / "08d_projections_year.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: 08d_projections_year.png")


# ── Save cluster assignment CSV ───────────────────────────────────────────────

out = df[["BirdID", "Year", "Sex", "Stage", "NumberIndependent",
          "SMIcntr", "Musclecntr", "Fatcntr",
          "PCV_percent_cntr", "Hbdiv10cntr"]].copy()
out[f"cluster_k{best_k}"] = km_best
out["PCA1"]  = Z_pca[:, 0]
out["PCA2"]  = Z_pca[:, 1]
out["tSNE1"] = Z_tsne[:, 0]
out["tSNE2"] = Z_tsne[:, 1]
out["UMAP1"] = Z_umap[:, 0]
out["UMAP2"] = Z_umap[:, 1]
out.to_csv(RESULTS / "08_dimred_clusters.csv", index=False)
print("Saved: 08_dimred_clusters.csv")

print(f"\nCluster summary (k={best_k}):")
for k in order:
    n_k = cluster_ns[k]
    mean_out = cluster_means[k]
    n_succ = (outcome[km_best == k] > 0).sum()
    print(f"  C{k+1}: n={n_k:3d}  mean_output={mean_out:.2f}  "
          f"success_rate={n_succ/n_k:.0%}")
print(f"\nKruskal-Wallis across clusters: H={kw_stat:.2f}, p={kw_p:.4f}")
print("\n[08_dimred_clustering] Done.\n")
