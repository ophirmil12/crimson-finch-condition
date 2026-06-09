"""
Step 07 – Sex × condition interaction with permutation tests.
Tests whether the SMI-reproduction relationship differs between males and females.
p-values for SMI coefficients and the interaction term are obtained via
permutation tests (B=1000) rather than relying on asymptotic GLM theory,
which can be unreliable for small within-group samples.
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
    COL_MALE, COL_FEMALE, COL_SMI,
    YEAR_MAP, YEAR_COLS, setup_style, ensure_dirs
)

setup_style()
ensure_dirs()

B = 1000   # permutation replicates
RNG = np.random.default_rng(42)


# ── Load data ─────────────────────────────────────────────────────────────────

df = pd.read_csv(POISSON_IND_FILE)
df["YearLabel"] = df["Year"].map({k: v[0] for k, v in YEAR_MAP.items()})
df = df.dropna(subset=["NumberIndependent", "SMIcntr", "SMIcntr2",
                        "Sex", "Stage", "Year"])
df["Stage"] = df["Stage"].astype("category")
df["Year"]  = df["Year"].astype("category")
df["Sex"]   = df["Sex"].astype(str)
df["SexLabel"] = df["Sex"].map({"0": "Female", "1": "Male"})

females = df[df["Sex"] == "0"].copy()
males   = df[df["Sex"] == "1"].copy()
print(f"N = {len(df)}  (females: {len(females)}, males: {len(males)})")


# ── AICc helper ───────────────────────────────────────────────────────────────

def aicc(fit, n):
    k = fit.df_model + 1
    return fit.aic + (2 * k * (k + 1)) / max(n - k - 1, 1)


# ── Permutation test helpers ──────────────────────────────────────────────────

def perm_pvalue_smi(data, formula, coef_name, b=B):
    """
    Permutation p-value for one GLM coefficient.
    Shuffles the predictor column named in coef_name, refits B times.
    Returns (observed_coef, null_distribution, two-sided p-value).
    """
    fit_obs = smf.glm(formula, data=data,
                      family=sm.families.Poisson()).fit(disp=False)
    obs = fit_obs.params[coef_name]

    null = np.empty(b)
    perm_data = data.copy()
    for i in range(b):
        perm_data[coef_name] = RNG.permutation(data[coef_name].values)
        try:
            fit_perm = smf.glm(formula, data=perm_data,
                               family=sm.families.Poisson()).fit(disp=False)
            null[i] = fit_perm.params[coef_name]
        except Exception:
            null[i] = 0.0

    p = (np.abs(null) >= np.abs(obs)).mean()
    return obs, null, p


def perm_pvalue_interaction(data, formula, coef_name, shuffle_col, b=B):
    """
    Permutation p-value for an interaction term.
    Shuffles shuffle_col (e.g. Sex) to break the sex-specific SMI relationship.
    Returns (observed_coef, null_distribution, two-sided p-value).
    """
    fit_obs = smf.glm(formula, data=data,
                      family=sm.families.Poisson()).fit(disp=False)
    obs = fit_obs.params[coef_name]

    null = np.empty(b)
    perm_data = data.copy()
    for i in range(b):
        perm_data[shuffle_col] = RNG.permutation(data[shuffle_col].values)
        try:
            fit_perm = smf.glm(formula, data=perm_data,
                               family=sm.families.Poisson()).fit(disp=False)
            null[i] = fit_perm.params.get(coef_name, 0.0)
        except Exception:
            null[i] = 0.0

    p = (np.abs(null) >= np.abs(obs)).mean()
    return obs, null, p


# ── Model comparison (AICc) ───────────────────────────────────────────────────

n = len(df)
model_formulas = {
    "Baseline (Stage + Year)":         "NumberIndependent ~ C(Stage) + C(Year)",
    "Additive (SMI + Sex)":            "NumberIndependent ~ SMIcntr + Sex + C(Stage) + C(Year)",
    "SMI only (no Sex)":               "NumberIndependent ~ SMIcntr + C(Stage) + C(Year)",
    "Sex only (no SMI)":               "NumberIndependent ~ Sex + C(Stage) + C(Year)",
    "Interaction (SMI x Sex)":         "NumberIndependent ~ SMIcntr * Sex + C(Stage) + C(Year)",
    "Interaction + quadratic SMI":     "NumberIndependent ~ SMIcntr * Sex + SMIcntr2 + C(Stage) + C(Year)",
}

fits = {}
for name, formula in model_formulas.items():
    try:
        fits[name] = smf.glm(formula, data=df,
                             family=sm.families.Poisson()).fit(disp=False)
    except Exception as e:
        print(f"  '{name}' failed: {e}")

comp = pd.DataFrame({
    name: {"AICc": aicc(fit, n), "deviance": fit.deviance,
           "n_params": fit.df_model + 1}
    for name, fit in fits.items()
}).T.sort_values("AICc")
comp["delta_AICc"] = comp["AICc"] - comp["AICc"].min()
comp["weight"] = np.exp(-0.5 * comp["delta_AICc"])
comp["weight"] /= comp["weight"].sum()

print("\n-- Model comparison (AICc)")
print(comp[["AICc", "delta_AICc", "weight", "n_params"]].round(3).to_string())
comp.round(4).to_csv(RESULTS / "07_sex_model_comparison.csv")


# ── Permutation tests ─────────────────────────────────────────────────────────

print(f"\n-- Running permutation tests (B={B}) ...")

# 1. SMI effect within females
print("   Permuting SMI within females ...")
obs_f, null_f, p_f = perm_pvalue_smi(
    females,
    "NumberIndependent ~ SMIcntr + C(Stage) + C(Year)",
    "SMIcntr"
)

# 2. SMI effect within males
print("   Permuting SMI within males ...")
obs_m, null_m, p_m = perm_pvalue_smi(
    males,
    "NumberIndependent ~ SMIcntr + C(Stage) + C(Year)",
    "SMIcntr"
)

# 3. Interaction term — shuffle Sex labels
print("   Permuting Sex labels for interaction term ...")
obs_int, null_int, p_int = perm_pvalue_interaction(
    df,
    "NumberIndependent ~ SMIcntr * Sex + C(Stage) + C(Year)",
    coef_name="SMIcntr:Sex[T.1]",
    shuffle_col="Sex"
)

print(f"\n-- Permutation results (B={B})")
print(f"   Female SMI  coef={obs_f:.3f}   perm-p={p_f:.3f}")
print(f"   Male   SMI  coef={obs_m:.3f}   perm-p={p_m:.3f}")
print(f"   Interaction  coef={obs_int:.3f}  perm-p={p_int:.3f}")

# Save permutation p-values alongside GLM p-values
int_fit   = fits["Interaction (SMI x Sex)"]
glm_p_f   = smf.glm("NumberIndependent ~ SMIcntr + C(Stage) + C(Year)",
                     data=females, family=sm.families.Poisson()).fit(disp=False).pvalues["SMIcntr"]
glm_p_m   = smf.glm("NumberIndependent ~ SMIcntr + C(Stage) + C(Year)",
                     data=males,   family=sm.families.Poisson()).fit(disp=False).pvalues["SMIcntr"]
glm_p_int = int_fit.pvalues["SMIcntr:Sex[T.1]"]

perm_summary = pd.DataFrame({
    "test":        ["Female SMI", "Male SMI", "Interaction (SMI x Sex)"],
    "obs_coef":    [obs_f, obs_m, obs_int],
    "IRR":         [np.exp(obs_f), np.exp(obs_m), np.exp(obs_int)],
    "GLM_p":       [glm_p_f, glm_p_m, glm_p_int],
    "perm_p":      [p_f, p_m, p_int],
    "B":           [B, B, B],
})
perm_summary.round(4).to_csv(RESULTS / "07_permutation_pvalues.csv", index=False)
print("\nSaved: 07_permutation_pvalues.csv")
print(perm_summary.to_string(index=False))


# ── Per-sex GLM fits (for figures) ────────────────────────────────────────────

sex_fits = {}
for label, subset in [("Female", females), ("Male", males)]:
    sex_fits[label] = smf.glm(
        "NumberIndependent ~ SMIcntr + C(Stage) + C(Year)",
        data=subset, family=sm.families.Poisson()
    ).fit(disp=False)

sex_summary = pd.DataFrame({
    label: {
        "n":           len(df[df["SexLabel"] == label]),
        "SMI_coef":    sex_fits[label].params["SMIcntr"],
        "SMI_se":      sex_fits[label].bse["SMIcntr"],
        "GLM_p":       sex_fits[label].pvalues["SMIcntr"],
        "perm_p":      p_f if label == "Female" else p_m,
        "SMI_IRR":     np.exp(sex_fits[label].params["SMIcntr"]),
        "mean_output": df[df["SexLabel"] == label]["NumberIndependent"].mean(),
    }
    for label in ["Female", "Male"]
}).T
sex_summary.round(4).to_csv(RESULTS / "07_per_sex_smi.csv")
print("Saved: 07_per_sex_smi.csv")


# ── Figure 1 – SMI vs output by sex (perm p-values in titles) ────────────────

fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharey=True)
fig.suptitle("SMI-reproduction relationship by sex", fontweight="bold")

smi_grid     = np.linspace(df["SMIcntr"].min(), df["SMIcntr"].max(), 200)
common_stage = df["Stage"].mode()[0]
common_year  = df["Year"].mode()[0]

perm_ps = {"Female": p_f, "Male": p_m}

for ax, (label, subset), col in zip(
    axes,
    [("Female", females), ("Male", males)],
    [COL_FEMALE, COL_MALE]
):
    jit = np.random.default_rng(42).uniform(-0.04, 0.04, size=len(subset))
    ax.scatter(subset["SMIcntr"] + jit, subset["NumberIndependent"],
               alpha=0.4, s=22, color=col, edgecolors="none", zorder=2)

    pred_df = pd.DataFrame({
        "SMIcntr": smi_grid,
        "Stage":   common_stage,
        "Year":    common_year,
    })
    ax.plot(smi_grid, sex_fits[label].predict(pred_df),
            color=col, linewidth=2.5, zorder=3)

    coef = sex_fits[label].params["SMIcntr"]
    irr  = np.exp(coef)
    pp   = perm_ps[label]
    ax.set_title(
        f"{label}  (n={len(subset)})\n"
        f"SMI coef={coef:.3f}, IRR={irr:.2f}\n"
        f"perm-p={pp:.3f}  (B={B})",
        fontsize=10
    )
    ax.set_xlabel("SMI (centred)")
    if ax is axes[0]:
        ax.set_ylabel("Number of independent young")

plt.tight_layout()
fig.savefig(FIGURES / "07a_smi_by_sex.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: 07a_smi_by_sex.png")


# ── Figure 2 – Permutation null distributions ─────────────────────────────────

fig, axes = plt.subplots(1, 3, figsize=(14, 4))
fig.suptitle(f"Permutation null distributions (B={B})", fontweight="bold")

for ax, null, obs, p, col, title in [
    (axes[0], null_f,   obs_f,   p_f,   COL_FEMALE, "Female SMI coef"),
    (axes[1], null_m,   obs_m,   p_m,   COL_MALE,   "Male SMI coef"),
    (axes[2], null_int, obs_int, p_int, COL_SMI,    "Interaction coef\n(SMI x Sex)"),
]:
    ax.hist(null, bins=40, color=col, alpha=0.7, edgecolor="white")
    ax.axvline( obs, color="black",  linewidth=2,   label=f"Observed ({obs:.3f})")
    ax.axvline(-obs, color="black",  linewidth=2,   linestyle="--")
    ax.axvline( np.percentile(null, 97.5), color="grey", linewidth=1,
               linestyle=":", label="95% null CI")
    ax.axvline( np.percentile(null, 2.5),  color="grey", linewidth=1, linestyle=":")
    ax.set_xlabel("Coefficient value")
    ax.set_ylabel("Count")
    ax.set_title(f"{title}\nperm-p = {p:.3f}")
    ax.legend(fontsize=8)

plt.tight_layout()
fig.savefig(FIGURES / "07e_permutation_distributions.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: 07e_permutation_distributions.png")


# ── Figure 3 – SMI tertile means by sex ──────────────────────────────────────

df["SMI_tertile"] = pd.qcut(df["SMIcntr"], q=3, labels=["Light", "Medium", "Heavy"])
tertile_sex = (df.groupby(["SexLabel", "SMI_tertile"], observed=True)["NumberIndependent"]
                 .mean().unstack("SMI_tertile"))

fig, ax = plt.subplots(figsize=(7, 4))
x = np.arange(3)
w = 0.35

for i, (sex_lbl, col) in enumerate([("Female", COL_FEMALE), ("Male", COL_MALE)]):
    vals   = tertile_sex.loc[sex_lbl, ["Light", "Medium", "Heavy"]].values.astype(float)
    offset = (i - 0.5) * w
    bars   = ax.bar(x + offset, vals, w, label=sex_lbl, color=col,
                    edgecolor="white", alpha=0.85)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.03,
                f"{v:.2f}", ha="center", fontsize=8)

ax.set_xticks(x)
ax.set_xticklabels(["Light", "Medium", "Heavy"])
ax.set_xlabel("SMI tertile")
ax.set_ylabel("Mean independent young")
ax.set_title("SMI tertile effect by sex")
ax.legend(loc="upper left")

# Ratio annotations placed below bars to avoid the legend
for i, sex_lbl in enumerate(["Female", "Male"]):
    row   = tertile_sex.loc[sex_lbl]
    ratio = row["Heavy"] / max(row["Light"], 1e-9)
    pp    = perm_ps[sex_lbl]
    ax.text(0.02 + i * 0.5, 0.04,
            f"{sex_lbl}: H/L={ratio:.1f}×  perm-p={pp:.3f}",
            transform=ax.transAxes, fontsize=9,
            color="black", va="bottom")

plt.tight_layout()
fig.savefig(FIGURES / "07b_tertile_by_sex.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: 07b_tertile_by_sex.png")


# ── Figure 4 – Forest: SMI by year and sex ────────────────────────────────────

year_sex_results = {}
for raw_year, (yr_label, yr_col) in YEAR_MAP.items():
    for sex_val, sex_label in [("0", "Female"), ("1", "Male")]:
        sub = df[(df["Year"] == raw_year) & (df["Sex"] == sex_val)]
        if len(sub) < 12:   # raised from 8 — below this, Stage dummies blow up SE
            continue
        try:
            fit = smf.glm("NumberIndependent ~ SMIcntr + C(Stage)",
                          data=sub, family=sm.families.Poisson()).fit(disp=False)
            coef = fit.params.get("SMIcntr", np.nan)
            se   = fit.bse.get("SMIcntr", np.nan)
            # skip entries where SE is implausibly large (numerical instability)
            if se > 5:
                continue
            year_sex_results[f"{yr_label} - {sex_label}"] = {
                "coef":  coef,
                "se":    se,
                "p":     fit.pvalues.get("SMIcntr", np.nan),
                "color": yr_col,
                "sex":   sex_label,
            }
        except Exception:
            pass

ys_df = pd.DataFrame(year_sex_results).T
coefs = ys_df["coef"].astype(float).values
ses   = ys_df["se"].astype(float).values
ps    = ys_df["p"].astype(float).values
face_colors = [COL_FEMALE if s == "Female" else COL_MALE for s in ys_df["sex"]]

fig, ax = plt.subplots(figsize=(8, max(4, len(ys_df) * 0.55)))
y_pos = np.arange(len(ys_df))
ax.barh(y_pos, coefs, xerr=1.96 * ses, color=face_colors,
        align="center", alpha=0.8, ecolor="#555", capsize=4,
        height=0.55, edgecolor="white")
ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
ax.set_yticks(y_pos)
ax.set_yticklabels(ys_df.index, fontsize=10)
ax.set_xlabel("SMI coefficient (± 1.96 SE)  [GLM p-values shown]")
ax.set_title("SMI effect by year and sex")
x_margin = max(np.abs(coefs) + 1.96 * ses) * 0.18
for i, (c, p, se) in enumerate(zip(coefs, ps, ses)):
    sig = "**" if p < 0.01 else ("*" if p < 0.05 else "ns")
    ax.text(c + 1.96 * se + x_margin * 0.3, i, sig, va="center", fontsize=9)
ax.legend(handles=[mpatches.Patch(color=COL_FEMALE, label="Female"),
                   mpatches.Patch(color=COL_MALE,   label="Male")],
          loc="lower right")
plt.tight_layout()
fig.savefig(FIGURES / "07c_forest_year_sex.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: 07c_forest_year_sex.png")


# ── Figure 5 – Model comparison ───────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(9, 4))
bar_cols = [COL_SMI if "x" in n.lower() else "#bbbbbb" for n in comp.index]
bars = ax.barh(comp.index, comp["delta_AICc"], color=bar_cols,
               edgecolor="white", alpha=0.85)
ax.axvline(2, color="black", linestyle="--", linewidth=0.8, label="delta_AICc = 2")
ax.set_xlabel("delta AICc")
ax.set_title("Model comparison -- Sex x SMI interaction")
ax.invert_yaxis()
ax.legend()
for bar, w in zip(bars, comp["weight"]):
    ax.text(bar.get_width() + 0.05, bar.get_y() + bar.get_height() / 2,
            f"w={w:.2f}", va="center", fontsize=9)
plt.tight_layout()
fig.savefig(FIGURES / "07d_model_comparison.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: 07d_model_comparison.png")

print("\n[07_sex_interaction] Done.\n")
