"""
Step 04 – Key visualizations.
1. PC2 (energy) vs number of independent young — the paper's main figure.
2. Tertile comparison: light vs heavy birds (3× effect).
3. Energy vs haematological indices compared side-by-side.
4. Individual index scatter plots vs outcome.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import statsmodels.formula.api as smf
import statsmodels.api as sm
from scipy.stats import pearsonr, mannwhitneyu

from config import (
    POISSON_IND_FILE, FIGURES, RESULTS,
    ENERGY_VARS, HAEM_VARS, ENERGY_COLORS, HAEM_COLORS,
    ENERGY_LABELS, HAEM_LABELS,
    COL_PC1, COL_PC2, YEAR_COLS, YEAR_MAP, COL_SMI,
    setup_style, ensure_dirs
)

setup_style()
ensure_dirs()

# ── Load data ─────────────────────────────────────────────────────────────────

df = pd.read_csv(POISSON_IND_FILE)
df["YearLabel"] = df["Year"].map({k: v[0] for k, v in YEAR_MAP.items()})
df["YearIdx"]   = df["Year"].map(
    {"1(2006-07)": 0, "2(2007-08)": 1, "3(2008-09)": 2, "4(2009-10)": 3})
df = df.dropna(subset=["NumberIndependent", "PC1", "PC2", "SMIcntr",
                        "Musclecntr", "Fatcntr", "PCV_percent_cntr", "Hbdiv10cntr"])


# ── Figure 1 – PC2 (energy) vs reproductive output ───────────────────────────
# Jittered scatter + running mean + GLM fitted line

fig, ax = plt.subplots(figsize=(7, 5))

jitter = np.random.default_rng(42).uniform(-0.05, 0.05, size=len(df))
ax.scatter(df["PC2"] + jitter, df["NumberIndependent"],
           alpha=0.3, s=18, color=COL_PC2, edgecolors="none", zorder=2)

# Fit smooth prediction line from a Poisson GLM (linear + quadratic terms)
fit_quad = smf.glm(
    "NumberIndependent ~ PC2 + PC2sq + C(Stage) + C(Year)",
    data=df, family=sm.families.Poisson()
).fit(disp=False)

x_range = np.linspace(df["PC2"].min(), df["PC2"].max(), 200)
# Predict at mean Stage and Year = most common
pred_df = pd.DataFrame({
    "PC2": x_range,
    "PC2sq": x_range ** 2,
    "Stage": df["Stage"].mode()[0],
    "Year": df["Year"].mode()[0],
})
y_pred = fit_quad.predict(pred_df)
ax.plot(x_range, y_pred, color=COL_PC2, linewidth=2.5, zorder=3,
        label=f"Poisson GLM (quadratic)\np(PC2²) = {fit_quad.pvalues.get('PC2sq', np.nan):.3f}")

ax.set_xlabel("PC2 score (energy-reserve axis: muscle, fat, mass)")
ax.set_ylabel("Number of independent young")
ax.set_title("PC2 predicts reproductive output")
ax.legend(fontsize=9, loc="upper right")

plt.tight_layout()
fig.savefig(FIGURES / "04a_pc2_vs_output.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: 04a_pc2_vs_output.png")


# ── Figure 2 – PC1 (haematological) vs reproductive output (for contrast) ────

fig, axes = plt.subplots(1, 2, figsize=(11, 5))
fig.suptitle("PC axes vs independent young", fontweight="bold")

for ax, pc, col, title in [
    (axes[0], "PC1", COL_PC1, "PC1 — haematological (PCV, Hb)"),
    (axes[1], "PC2", COL_PC2, "PC2 — energy reserves (SMI, Muscle, Fat)"),
]:
    jitter = np.random.default_rng(0).uniform(-0.06, 0.06, size=len(df))
    ax.scatter(df[pc] + jitter, df["NumberIndependent"],
               alpha=0.3, s=18, color=col, edgecolors="none")

    r, p = pearsonr(df[pc], df["NumberIndependent"])
    ax.set_xlabel(f"{pc} score")
    ax.set_ylabel("Number of independent young")
    ax.set_title(f"{title}\nr = {r:.2f}, p = {p:.3f}")

    # Simple OLS trend line for visual only
    m, b = np.polyfit(df[pc], df["NumberIndependent"], 1)
    xs = np.linspace(df[pc].min(), df[pc].max(), 100)
    ax.plot(xs, m * xs + b, color=col, linewidth=1.8, linestyle="--")

plt.tight_layout()
fig.savefig(FIGURES / "04b_pc1_vs_pc2_output.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: 04b_pc1_vs_pc2_output.png")


# ── Figure 3 – Tertile box plot: light / medium / heavy birds ────────────────
# Reproduces the paper's core message: heavy birds ≈ 3× output of light birds

df["SMI_tertile"] = pd.qcut(df["SMIcntr"], q=3, labels=["Light", "Medium", "Heavy"])

tertile_means = df.groupby("SMI_tertile", observed=True)["NumberIndependent"].mean()
print("\nMean independent young by SMI tertile:")
print(tertile_means.round(3))
ratio = tertile_means["Heavy"] / tertile_means["Light"]
print(f"Heavy / Light ratio: {ratio:.2f}x")

fig, axes = plt.subplots(1, 2, figsize=(11, 5))
fig.suptitle("Scaled Mass Index (SMI) tertiles and reproductive output", fontweight="bold")

tertile_colors = [YEAR_COLS[0], YEAR_COLS[1], COL_SMI]

ax = axes[0]
groups = [df[df["SMI_tertile"] == t]["NumberIndependent"].values
          for t in ["Light", "Medium", "Heavy"]]
bp = ax.boxplot(groups, patch_artist=True, notch=False,
                medianprops=dict(color="white", linewidth=2))
for patch, col in zip(bp["boxes"], tertile_colors):
    patch.set_facecolor(col)
    patch.set_alpha(0.8)
ax.set_xticklabels(["Light", "Medium", "Heavy"])
ax.set_xlabel("SMI tertile")
ax.set_ylabel("Number of independent young")
ax.set_title("Distribution by body-condition tertile")
ax.text(0.97, 0.97, f"{ratio:.1f}× more young\n(Heavy vs Light)",
        transform=ax.transAxes, ha="right", va="top", fontsize=10,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", edgecolor="gray"))

ax = axes[1]
ax.bar(["Light", "Medium", "Heavy"], tertile_means.values,
       color=tertile_colors, edgecolor="white", alpha=0.85)
ax.set_xlabel("SMI tertile")
ax.set_ylabel("Mean independent young")
ax.set_title("Mean reproductive output by tertile")
for i, (lbl, val) in enumerate(zip(["Light", "Medium", "Heavy"], tertile_means.values)):
    ax.text(i, val + 0.03, f"{val:.2f}", ha="center", fontsize=10)

plt.tight_layout()
fig.savefig(FIGURES / "04c_smi_tertile.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: 04c_smi_tertile.png")


# ── Figure 4 – Energy vs haematological indices compared ─────────────────────
# Shows all 5 individual indices vs output in a grid

all_vars   = ENERGY_VARS + HAEM_VARS
all_labels = ENERGY_LABELS + HAEM_LABELS
all_colors = ENERGY_COLORS + HAEM_COLORS
group_label = (["Energy"] * 3) + (["Haematological"] * 2)

fig, axes = plt.subplots(1, 5, figsize=(16, 4), sharey=True)
fig.suptitle("Individual condition indices vs independent young", fontweight="bold")

for ax, var, lbl, col, grp in zip(axes, all_vars, all_labels, all_colors, group_label):
    jit = np.random.default_rng(1).uniform(-0.05, 0.05, size=len(df))
    ax.scatter(df[var] + jit, df["NumberIndependent"],
               alpha=0.3, s=14, color=col, edgecolors="none")
    m, b = np.polyfit(df[var], df["NumberIndependent"], 1)
    xs = np.linspace(df[var].min(), df[var].max(), 100)
    ax.plot(xs, m * xs + b, color=col, linewidth=2)
    r, p = pearsonr(df[var], df["NumberIndependent"])
    ax.set_title(f"{lbl}\nr={r:.2f}, p={p:.3f}", fontsize=9)
    ax.set_xlabel("Centred value")
    if ax is axes[0]:
        ax.set_ylabel("Independent young")
    # Shade background by group
    bg = "#eaf3fb" if grp == "Energy" else "#fdf0e8"
    ax.set_facecolor(bg)
    ax.text(0.05, 0.97, grp, transform=ax.transAxes, fontsize=7,
            va="top", color="gray")

plt.tight_layout()
fig.savefig(FIGURES / "04d_all_indices_vs_output.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: 04d_all_indices_vs_output.png")


# ── Figure 5 – SMI: zero fledglings vs any fledglings ────────────────────────

B_PERM = 1000
RNG04  = np.random.default_rng(42)

df["outcome"] = np.where(df["NumberIndependent"] > 0, ">0 fledglings", "0 fledglings")

grp_zero = df.loc[df["outcome"] == "0 fledglings",  "SMIcntr"]
grp_any  = df.loc[df["outcome"] == ">0 fledglings", "SMIcntr"]

# Observed mean difference (any − zero)
obs_diff = grp_any.mean() - grp_zero.mean()

# Permutation test: shuffle outcome labels, recompute mean difference
all_smi  = df["SMIcntr"].values
n_any    = len(grp_any)
null_diffs = np.empty(B_PERM)
for i in range(B_PERM):
    perm = RNG04.permutation(all_smi)
    null_diffs[i] = perm[:n_any].mean() - perm[n_any:].mean()
p_perm = (null_diffs >= obs_diff).mean()

pool_std = np.sqrt((grp_any.std() ** 2 + grp_zero.std() ** 2) / 2)
cohens_d = obs_diff / pool_std

print(f"\nSMI: 0 vs >0 fledglings")
print(f"  n(zero)={len(grp_zero)}, mean SMI={grp_zero.mean():.3f}  sd={grp_zero.std():.3f}")
print(f"  n(any) ={len(grp_any)},  mean SMI={grp_any.mean():.3f}  sd={grp_any.std():.3f}")
print(f"  Observed mean diff={obs_diff:.3f}")
print(f"  Permutation p={p_perm:.4f}  (B={B_PERM}, one-sided)")
print(f"  Cohen's d={cohens_d:.3f}")

fig, axes = plt.subplots(1, 2, figsize=(10, 5))
fig.suptitle("SMI in birds that failed vs succeeded\n(0 independent young vs ≥1)",
             fontweight="bold")

groups   = ["0 fledglings", ">0 fledglings"]
grp_data = [grp_zero.values, grp_any.values]
grp_cols = [YEAR_COLS[0], COL_SMI]

# Panel A – boxplot
ax = axes[0]
bp = ax.boxplot(grp_data, patch_artist=True, notch=False,
                medianprops=dict(color="white", linewidth=2.5),
                whiskerprops=dict(linewidth=1.2),
                capprops=dict(linewidth=1.2))
for patch, col in zip(bp["boxes"], grp_cols):
    patch.set_facecolor(col)
    patch.set_alpha(0.8)
rng = np.random.default_rng(7)
for i, (data, col) in enumerate(zip(grp_data, grp_cols), start=1):
    jit = rng.uniform(-0.12, 0.12, size=len(data))
    ax.scatter(np.full(len(data), i) + jit, data,
               alpha=0.35, s=18, color=col, edgecolors="none", zorder=3)
ax.set_xticks([1, 2])
ax.set_xticklabels(groups)
ax.set_ylabel("SMI (centred)")
ax.set_title("Distribution of SMI by breeding outcome")
ax.text(0.97, 0.97,
        f"perm-p={p_perm:.4f}\n(B={B_PERM}, one-sided)\nd={cohens_d:.2f}",
        transform=ax.transAxes, ha="right", va="top", fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", edgecolor="gray"))

# Panel B – histogram overlay
ax = axes[1]
bins = np.linspace(df["SMIcntr"].min(), df["SMIcntr"].max(), 25)
for data, col, lbl in zip(grp_data, grp_cols, groups):
    ax.hist(data, bins=bins, color=col, alpha=0.55, label=lbl,
            edgecolor="white", density=True)
    ax.axvline(data.mean(), color=col, linewidth=2, linestyle="--")
ax.set_xlabel("SMI (centred)")
ax.set_ylabel("Density")
ax.set_title("SMI distributions (dashed = group mean)")
ax.legend(fontsize=9)

plt.tight_layout()
fig.savefig(FIGURES / "04e_smi_zero_vs_any.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: 04e_smi_zero_vs_any.png")

print("\n[04_visualization] Done.\n")
