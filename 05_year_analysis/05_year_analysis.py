"""
Step 05 – Year-consistency analysis.
Fits per-year Poisson GLMs and checks whether the SMI / PC2 effect is stable
or varies across the four breeding seasons (as noted in the paper).
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

from config import (
    POISSON_IND_FILE, FIGURES, RESULTS,
    ENERGY_VARS, HAEM_VARS, ENERGY_COLORS, HAEM_COLORS,
    ENERGY_LABELS, HAEM_LABELS,
    COL_PC1, COL_PC2, COL_SMI, YEAR_COLS, YEAR_MAP,
    YEAR_LABELS, setup_style, ensure_dirs
)

setup_style()
ensure_dirs()


# ── Load data ─────────────────────────────────────────────────────────────────

df = pd.read_csv(POISSON_IND_FILE)
df["YearLabel"] = df["Year"].map({k: v[0] for k, v in YEAR_MAP.items()})
df["YearIdx"]   = df["Year"].map(
    {"1(2006-07)": 0, "2(2007-08)": 1, "3(2008-09)": 2, "4(2009-10)": 3})
df = df.dropna(subset=["NumberIndependent", "PC1", "PC2", "SMIcntr",
                        "Musclecntr", "Fatcntr", "PCV_percent_cntr",
                        "Hbdiv10cntr", "Stage"])
df["Stage"] = df["Stage"].astype("category")
df["Sex"]   = df["Sex"].astype(str)


# ── Per-year GLMs ─────────────────────────────────────────────────────────────

year_results = {}
formula_pc  = "NumberIndependent ~ PC2 + C(Stage)"
formula_smi = "NumberIndependent ~ SMIcntr + C(Stage)"

for raw_year, (label, col) in YEAR_MAP.items():
    sub = df[df["Year"] == raw_year].copy()
    if len(sub) < 10:
        print(f"  Skipping {label}: only {len(sub)} records")
        continue
    try:
        fit_pc  = smf.glm(formula_pc,  data=sub, family=sm.families.Poisson()).fit(disp=False)
        fit_smi = smf.glm(formula_smi, data=sub, family=sm.families.Poisson()).fit(disp=False)
        year_results[label] = {
            "n": len(sub),
            "color": col,
            "mean_indep": sub["NumberIndependent"].mean(),
            # PC2 model
            "PC2_coef":  fit_pc.params.get("PC2", np.nan),
            "PC2_se":    fit_pc.bse.get("PC2", np.nan),
            "PC2_p":     fit_pc.pvalues.get("PC2", np.nan),
            "PC2_IRR":   np.exp(fit_pc.params.get("PC2", np.nan)),
            # SMI model
            "SMI_coef":  fit_smi.params.get("SMIcntr", np.nan),
            "SMI_se":    fit_smi.bse.get("SMIcntr", np.nan),
            "SMI_p":     fit_smi.pvalues.get("SMIcntr", np.nan),
            "SMI_IRR":   np.exp(fit_smi.params.get("SMIcntr", np.nan)),
        }
        print(f"  {label} (n={len(sub)}): PC2 coef={fit_pc.params.get('PC2',np.nan):.3f} "
              f"(p={fit_pc.pvalues.get('PC2',np.nan):.3f}), "
              f"SMI coef={fit_smi.params.get('SMIcntr',np.nan):.3f} "
              f"(p={fit_smi.pvalues.get('SMIcntr',np.nan):.3f})")
    except Exception as e:
        print(f"  {label} failed: {e}")

res_df = pd.DataFrame(year_results).T
res_df.to_csv(RESULTS / "05_per_year_effects.csv")
print("\nSaved: 05_per_year_effects.csv")
print(res_df[["n", "mean_indep", "PC2_coef", "PC2_p", "SMI_coef", "SMI_p",
              "PC2_IRR", "SMI_IRR"]].round(3).to_string())


# ── Figure 1 – Per-year coefficient forest plot ──────────────────────────────

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
fig.suptitle("Per-year Poisson GLM coefficients (log scale) — year-dependence check",
             fontweight="bold")

for ax, metric, col_coef, col_se, title in [
    (axes[0], "PC2", "PC2_coef", "PC2_se", "PC2 (energy-reserve axis)"),
    (axes[1], "SMI", "SMI_coef", "SMI_se", "Scaled Mass Index (SMI)"),
]:
    years  = list(res_df.index)
    coefs  = res_df[col_coef].astype(float).values
    ses    = res_df[col_se].astype(float).values
    ps     = res_df[f"{metric}_p"].astype(float).values
    colors = [yr_res["color"] for yr_res in year_results.values()]

    y_pos = np.arange(len(years))
    ax.barh(y_pos, coefs, xerr=1.96 * ses, color=colors,
            align="center", alpha=0.85, ecolor="#555555", capsize=4,
            height=0.5, edgecolor="white")
    ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(years)
    ax.set_xlabel("GLM coefficient (log scale)\n± 1.96 SE")
    ax.set_title(title)
    for i, (c, p) in enumerate(zip(coefs, ps)):
        sig = "**" if p < 0.01 else ("*" if p < 0.05 else "ns")
        ax.text(c + 1.96 * ses[i] + 0.01, i, sig, va="center", fontsize=9)

plt.tight_layout()
fig.savefig(FIGURES / "05a_per_year_forest.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: 05a_per_year_forest.png")


# ── Figure 2 – Mean output and SMI effect by year ────────────────────────────

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
fig.suptitle("Year-to-year variation in reproductive output and condition effect",
             fontweight="bold")

ax = axes[0]
colors = [yr_res["color"] for yr_res in year_results.values()]
ax.bar(list(res_df.index), res_df["mean_indep"].astype(float),
       color=colors, edgecolor="white", alpha=0.85)
ax.set_xlabel("Breeding season")
ax.set_ylabel("Mean independent young per bird")
ax.set_title("Mean reproductive output per year")
for i, (yr, val) in enumerate(zip(res_df.index, res_df["mean_indep"].astype(float))):
    ax.text(i, val + 0.02, f"{val:.2f}", ha="center", fontsize=9)

ax = axes[1]
ax.scatter(res_df["mean_indep"].astype(float),
           res_df["SMI_IRR"].astype(float),
           c=colors, s=120, zorder=3, edgecolors="white", linewidths=1.5)
for yr, row in res_df.iterrows():
    ax.annotate(yr, (float(row["mean_indep"]), float(row["SMI_IRR"])),
                textcoords="offset points", xytext=(6, 4), fontsize=9)
ax.axhline(1.0, color="black", linestyle="--", linewidth=0.8)
ax.set_xlabel("Mean independent young (year quality)")
ax.set_ylabel("SMI Incidence Rate Ratio (exp[coef])")
ax.set_title("SMI effect stronger in productive years?")

plt.tight_layout()
fig.savefig(FIGURES / "05b_year_variation.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: 05b_year_variation.png")


# ── Figure 3 – SMI tertile means per year (within-year tertiles) ─────────────
# Tertiles computed separately within each year, matching the paper's approach.
# This is what produces the ~3x ratio in 2009-10.

chunks = []
for _, grp in df.groupby("YearLabel"):
    grp = grp.copy()
    try:
        grp["SMI_tertile"] = pd.qcut(grp["SMIcntr"], q=3,
                                      labels=["Light", "Medium", "Heavy"])
    except ValueError:
        grp["SMI_tertile"] = np.nan
    chunks.append(grp)
df = pd.concat(chunks)

year_tertile = (df.groupby(["YearLabel", "SMI_tertile"], observed=True)["NumberIndependent"]
                  .mean().unstack("SMI_tertile"))

# Compute and print within-year Heavy/Light ratios
print("\nWithin-year Heavy/Light ratios:")
ratios = {}
for yr in year_tertile.index:
    light = year_tertile.loc[yr, "Light"]
    heavy = year_tertile.loc[yr, "Heavy"]
    ratio = heavy / light if light > 0 else float("inf")
    ratios[yr] = ratio
    print(f"  {yr}: Light={light:.2f}, Heavy={heavy:.2f}, ratio={ratio:.2f}x")

fig, ax = plt.subplots(figsize=(9, 5))
x = np.arange(len(year_tertile))
w = 0.25
tertile_colors = [YEAR_COLS[0], YEAR_COLS[1], COL_SMI]

for i, (tertile, col) in enumerate(zip(["Light", "Medium", "Heavy"], tertile_colors)):
    vals = year_tertile[tertile].values
    bars = ax.bar(x + i * w, vals, w, label=tertile, color=col,
                  edgecolor="white", alpha=0.85)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.04,
                f"{v:.2f}", ha="center", fontsize=7.5)

# Compute a consistent annotation height above all bars + leave title room
bar_top = float(np.nanmax(year_tertile.values))
annot_y = bar_top * 1.08
ax.set_ylim(0, bar_top * 1.38)   # extra headroom for annotations + legend

for j, yr in enumerate(year_tertile.index):
    r = ratios[yr]
    lbl = f"{r:.1f}×" if np.isfinite(r) else "n/a"
    ax.text(x[j] + w, annot_y, f"H/L={lbl}", ha="center", fontsize=9,
            fontweight="bold", color=COL_SMI)

ax.set_xticks(x + w)
ax.set_xticklabels(year_tertile.index)
ax.set_xlabel("Breeding season")
ax.set_ylabel("Mean independent young")
ax.set_title("SMI tertile effect by year (within-year tertiles)")
ax.legend(title="SMI tertile", loc="upper left")

plt.tight_layout()
fig.savefig(FIGURES / "05c_tertile_by_year.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: 05c_tertile_by_year.png")

print("\n[05_year_analysis] Done.\n")
