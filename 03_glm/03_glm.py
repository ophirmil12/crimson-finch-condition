"""
Step 03 – Poisson GLM: independent young ~ PC scores + covariates.
Reproduces the paper's model selection approach using AICc.
Exports coefficient tables and model comparison.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
import statsmodels.api as sm

from config import (
    POISSON_IND_FILE, FIGURES, RESULTS,
    COL_PC1, COL_PC2, YEAR_COLS, YEAR_MAP, setup_style, ensure_dirs
)

setup_style()
ensure_dirs()


# ── Load & prepare data ───────────────────────────────────────────────────────

df = pd.read_csv(POISSON_IND_FILE)
df["YearLabel"] = df["Year"].map({k: v[0] for k, v in YEAR_MAP.items()})
df["SexLabel"]  = df["Sex"].map({0: "Female", 1: "Male"})

# Drop rows missing key predictors or response
cols_needed = ["NumberIndependent", "PC1", "PC2", "Sex", "Stage", "Year",
               "PC1sq", "PC2sq", "SMIcntr", "Musclecntr", "Fatcntr",
               "PCV_percent_cntr", "Hbdiv10cntr"]
df = df.dropna(subset=cols_needed).copy()
print(f"Working dataset: {len(df)} records after dropping missing values")

# Convert categorical variables for formula API
df["Sex"]   = df["Sex"].astype(str)       # "0" / "1"
df["Stage"] = df["Stage"].astype("category")
df["Year"]  = df["Year"].astype("category")


# ── AICc helper ──────────────────────────────────────────────────────────────

def aicc(model_result, n):
    """Small-sample corrected AIC for a fitted statsmodels GLM result."""
    k = model_result.df_model + 1   # number of estimated parameters incl. intercept
    aic = model_result.aic
    correction = (2 * k * (k + 1)) / (n - k - 1) if n - k - 1 > 0 else np.inf
    return aic + correction


# ── Candidate model set (mirrors paper's six families) ───────────────────────
#
# Baseline:  intercept + Stage + Year  (structural covariates)
# +condition linear:   + PC1 + PC2
# +condition quadratic: + PC1 + PC2 + PC1sq + PC2sq
# +sex:      any of the above + Sex
#
# We fit all combinations and rank by AICc.

n = len(df)

candidate_formulas = {
    "Baseline":               "NumberIndependent ~ C(Stage) + C(Year)",
    "Base + Sex":             "NumberIndependent ~ C(Stage) + C(Year) + Sex",
    "Base + PC(lin)":         "NumberIndependent ~ C(Stage) + C(Year) + PC1 + PC2",
    "Base + PC(lin) + Sex":   "NumberIndependent ~ C(Stage) + C(Year) + PC1 + PC2 + Sex",
    "Base + PC(quad)":        "NumberIndependent ~ C(Stage) + C(Year) + PC1 + PC2 + PC1sq + PC2sq",
    "Base + PC(quad) + Sex":  "NumberIndependent ~ C(Stage) + C(Year) + PC1 + PC2 + PC1sq + PC2sq + Sex",
    "Base + SMI only":        "NumberIndependent ~ C(Stage) + C(Year) + SMIcntr",
    "Base + Energy indices":  "NumberIndependent ~ C(Stage) + C(Year) + SMIcntr + Musclecntr + Fatcntr",
    "Base + Haem indices":    "NumberIndependent ~ C(Stage) + C(Year) + PCV_percent_cntr + Hbdiv10cntr",
    "Base + All indices":     "NumberIndependent ~ C(Stage) + C(Year) + SMIcntr + Musclecntr + Fatcntr + PCV_percent_cntr + Hbdiv10cntr",
}

results = {}
for name, formula in candidate_formulas.items():
    try:
        fit = smf.glm(formula, data=df,
                      family=sm.families.Poisson()).fit(disp=False)
        results[name] = {
            "fit":    fit,
            "AICc":   aicc(fit, n),
            "AIC":    fit.aic,
            "deviance": fit.deviance,
            "df_resid": fit.df_resid,
            "n_params": fit.df_model + 1,
        }
    except Exception as e:
        print(f"  Model '{name}' failed: {e}")

# ── Model comparison table ────────────────────────────────────────────────────

comp_df = pd.DataFrame(
    {k: {kk: vv for kk, vv in v.items() if kk != "fit"} for k, v in results.items()}
).T.sort_values("AICc")

comp_df["delta_AICc"] = comp_df["AICc"] - comp_df["AICc"].min()
comp_df["weight"] = np.exp(-0.5 * comp_df["delta_AICc"])
comp_df["weight"] /= comp_df["weight"].sum()

print("\n── Model comparison (ranked by AICc) ──")
print(comp_df[["AICc", "delta_AICc", "weight", "deviance", "n_params"]].round(3).to_string())
comp_df.drop(columns=["fit"], errors="ignore").round(4).to_csv(RESULTS / "03_model_comparison.csv")
print("Saved: 03_model_comparison.csv")


# ── Best model summary ────────────────────────────────────────────────────────

best_name = comp_df.index[0]
best_fit  = results[best_name]["fit"]
print(f"\n── Best model: '{best_name}' (ΔAICc = 0.0) ──")
print(best_fit.summary())

# Save coefficient table
coef_df = pd.DataFrame({
    "coef":    best_fit.params,
    "se":      best_fit.bse,
    "z":       best_fit.tvalues,
    "p":       best_fit.pvalues,
    "CI_low":  best_fit.conf_int()[0],
    "CI_high": best_fit.conf_int()[1],
    "IRR":     np.exp(best_fit.params),       # incidence-rate ratio
})
coef_df.round(4).to_csv(RESULTS / "03_best_model_coefs.csv")
print("Saved: 03_best_model_coefs.csv")


# ── Also print PC-only model for comparison ───────────────────────────────────

pc_name = "Base + PC(lin)"
if pc_name in results:
    print(f"\n── PC linear model: '{pc_name}' ──")
    pc_fit = results[pc_name]["fit"]
    pc_coef = pd.DataFrame({
        "coef": pc_fit.params, "se": pc_fit.bse,
        "p": pc_fit.pvalues, "IRR": np.exp(pc_fit.params),
    })
    print(pc_coef[pc_coef.index.isin(["PC1", "PC2"])].round(4))


# ── Figure – AICc delta weights ──────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(10, 4))
colors = [COL_PC2 if "PC" in n else "#aaaaaa" for n in comp_df.index]
bars = ax.barh(comp_df.index, comp_df["delta_AICc"], color=colors, edgecolor="white")
ax.axvline(2, color="black", linestyle="--", linewidth=0.8, label="ΔAICc = 2")
ax.set_xlabel("ΔAICc (vs best model)")
ax.set_title("Poisson GLM model comparison — independent young")
ax.invert_yaxis()
ax.legend()

for bar, w in zip(bars, comp_df["weight"]):
    ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height() / 2,
            f"w={w:.2f}", va="center", fontsize=8)

plt.tight_layout()
fig.savefig(FIGURES / "03_model_comparison.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: 03_model_comparison.png")


# ── Predicted vs observed ─────────────────────────────────────────────────────

df["predicted"] = best_fit.predict(df)

fig, ax = plt.subplots(figsize=(5, 4))
ax.scatter(df["predicted"], df["NumberIndependent"],
           alpha=0.3, s=20, color=COL_PC2, edgecolors="none")
lims = [0, max(df["predicted"].max(), df["NumberIndependent"].max()) + 0.5]
ax.plot(lims, lims, "k--", linewidth=1, label="Perfect fit")
ax.set_xlabel("Predicted (Poisson GLM)")
ax.set_ylabel("Observed")
ax.set_title(f"Best model: {best_name}")
ax.legend(fontsize=9, loc="upper left")

plt.tight_layout()
fig.savefig(FIGURES / "03b_predicted_vs_observed.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: 03b_predicted_vs_observed.png")

print("\n[03_glm] Done.\n")
