"""
Step 06 – Basic ML comparison.
Can we predict independent young better without PCA, using raw condition indices?
Models: Poisson GLM (baseline), Ridge Poisson, Random Forest, Gradient Boosting.
Evaluation: cross-validated MAE, RMSE, and Poisson deviance.
Feature importance shows which indices matter most without PCA compression.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings
warnings.filterwarnings("ignore")
from scipy.ndimage import gaussian_filter1d

from sklearn.model_selection import cross_val_score, KFold, cross_validate
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import PoissonRegressor, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.inspection import permutation_importance
import statsmodels.formula.api as smf
import statsmodels.api as sm

from config import (
    POISSON_IND_FILE, FIGURES, RESULTS,
    ENERGY_VARS, HAEM_VARS, ENERGY_COLORS, HAEM_COLORS,
    ENERGY_LABELS, HAEM_LABELS,
    COL_PC1, COL_PC2, COL_SMI, YEAR_COLS, YEAR_MAP,
    setup_style, ensure_dirs
)

setup_style()
ensure_dirs()


# ── Load & prepare features ───────────────────────────────────────────────────

df = pd.read_csv(POISSON_IND_FILE)
df["YearLabel"] = df["Year"].map({k: v[0] for k, v in YEAR_MAP.items()})
df = df.dropna(subset=["NumberIndependent", "PC1", "PC2",
                        "SMIcntr", "Musclecntr", "Fatcntr",
                        "PCV_percent_cntr", "Hbdiv10cntr"])

# Encode categorical covariates
df["Sex_enc"]   = df["Sex"].astype(int)
df["Stage_enc"] = df["Stage"].map({"pre": 0, "I": 1, "N": 2, "post": 3}).fillna(2)
df["Year_enc"]  = df["Year"].map(
    {"1(2006-07)": 0, "2(2007-08)": 1, "3(2008-09)": 2, "4(2009-10)": 3})

y = df["NumberIndependent"].values

# Feature sets to compare
FEAT_PC = ["PC1", "PC2", "Sex_enc", "Stage_enc", "Year_enc"]
FEAT_RAW_COND = (ENERGY_VARS + HAEM_VARS +
                 ["Sex_enc", "Stage_enc", "Year_enc"])
FEAT_RAW_COND_LABELS = (ENERGY_LABELS + HAEM_LABELS +
                        ["Sex", "Stage", "Year"])
FEAT_ENERGY_ONLY = (ENERGY_VARS + ["Sex_enc", "Stage_enc", "Year_enc"])
FEAT_HAEM_ONLY   = (HAEM_VARS   + ["Sex_enc", "Stage_enc", "Year_enc"])


# ── Cross-validation helpers ──────────────────────────────────────────────────

cv = KFold(n_splits=5, shuffle=True, random_state=42)

def cv_scores(model, X, y):
    """Return mean MAE, RMSE across 5-fold CV."""
    mae_scores, rmse_scores = [], []
    for train_idx, test_idx in cv.split(X):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]
        model.fit(X_tr, y_tr)
        pred = np.maximum(model.predict(X_te), 0)   # counts can't be negative
        mae_scores.append(mean_absolute_error(y_te, pred))
        rmse_scores.append(np.sqrt(mean_squared_error(y_te, pred)))
    return np.mean(mae_scores), np.mean(rmse_scores)


def poisson_deviance(y_true, y_pred):
    """Mean Poisson deviance (lower = better)."""
    y_pred = np.maximum(y_pred, 1e-9)
    return 2 * np.mean(y_true * np.log(np.maximum(y_true, 1e-9) / y_pred)
                       - (y_true - y_pred))


# ── Build pipelines ───────────────────────────────────────────────────────────

models = {
    "Poisson GLM (PC scores)": Pipeline([
        ("scaler", StandardScaler()),
        ("model", PoissonRegressor(alpha=0, max_iter=500)),
    ]),
    "Poisson GLM (raw indices)": Pipeline([
        ("scaler", StandardScaler()),
        ("model", PoissonRegressor(alpha=0, max_iter=500)),
    ]),
    "Ridge Poisson (raw indices)": Pipeline([
        ("scaler", StandardScaler()),
        ("model", PoissonRegressor(alpha=1.0, max_iter=500)),
    ]),
    "Random Forest (raw indices)": Pipeline([
        ("scaler", StandardScaler()),
        ("model", RandomForestRegressor(n_estimators=200, max_depth=5,
                                        random_state=42, n_jobs=-1)),
    ]),
    "Gradient Boosting (raw indices)": Pipeline([
        ("scaler", StandardScaler()),
        ("model", GradientBoostingRegressor(n_estimators=200, max_depth=3,
                                            learning_rate=0.05, random_state=42)),
    ]),
    "Poisson GLM (energy only)": Pipeline([
        ("scaler", StandardScaler()),
        ("model", PoissonRegressor(alpha=0, max_iter=500)),
    ]),
    "Poisson GLM (haem only)": Pipeline([
        ("scaler", StandardScaler()),
        ("model", PoissonRegressor(alpha=0, max_iter=500)),
    ]),
}

feat_map = {
    "Poisson GLM (PC scores)":          df[FEAT_PC].values,
    "Poisson GLM (raw indices)":        df[FEAT_RAW_COND].values,
    "Ridge Poisson (raw indices)":      df[FEAT_RAW_COND].values,
    "Random Forest (raw indices)":      df[FEAT_RAW_COND].values,
    "Gradient Boosting (raw indices)":  df[FEAT_RAW_COND].values,
    "Poisson GLM (energy only)":        df[FEAT_ENERGY_ONLY].values,
    "Poisson GLM (haem only)":          df[FEAT_HAEM_ONLY].values,
}

bar_colors = {
    "Poisson GLM (PC scores)":          COL_PC2,
    "Poisson GLM (raw indices)":        HAEM_COLORS[0],
    "Ridge Poisson (raw indices)":      HAEM_COLORS[1],
    "Random Forest (raw indices)":      ENERGY_COLORS[0],
    "Gradient Boosting (raw indices)":  ENERGY_COLORS[1],
    "Poisson GLM (energy only)":        COL_SMI,
    "Poisson GLM (haem only)":          COL_PC1,
}

# ── Run CV ────────────────────────────────────────────────────────────────────

cv_results = {}
print("Running 5-fold cross-validation …")
for name, model in models.items():
    X = feat_map[name]
    mae, rmse = cv_scores(model, X, y)
    cv_results[name] = {"MAE": mae, "RMSE": rmse}
    print(f"  {name:<42} MAE={mae:.3f}  RMSE={rmse:.3f}")

cv_df = pd.DataFrame(cv_results).T.sort_values("MAE")
cv_df.to_csv(RESULTS / "06_ml_cv_scores.csv")
print("\nSaved: 06_ml_cv_scores.csv")
print(cv_df.round(3).to_string())


# ── Figure 1 – CV performance comparison ─────────────────────────────────────

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("ML model comparison — predicting number of independent young\n"
             "(5-fold cross-validation)", fontweight="bold")

for ax, metric in zip(axes, ["MAE", "RMSE"]):
    vals   = cv_df[metric].values
    names  = cv_df.index.tolist()
    colors = [bar_colors[n] for n in names]
    bars = ax.barh(names, vals, color=colors, edgecolor="white", alpha=0.85)
    ax.set_xlabel(metric)
    ax.set_title(f"5-fold CV {metric} (lower = better)")
    ax.invert_yaxis()
    for bar, v in zip(bars, vals):
        ax.text(v + 0.003, bar.get_y() + bar.get_height() / 2,
                f"{v:.3f}", va="center", fontsize=8)

plt.tight_layout()
fig.savefig(FIGURES / "06a_ml_comparison.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: 06a_ml_comparison.png")


# ── Figure 2 – Random Forest feature importances ─────────────────────────────

rf_pipe = models["Random Forest (raw indices)"]
rf_pipe.fit(df[FEAT_RAW_COND].values, y)
rf_model = rf_pipe.named_steps["model"]

importances = rf_model.feature_importances_
imp_df = pd.DataFrame({
    "feature": FEAT_RAW_COND_LABELS,
    "importance": importances,
    "color": (ENERGY_COLORS + HAEM_COLORS + ["#aaaaaa", "#aaaaaa", "#aaaaaa"]),
}).sort_values("importance", ascending=True)

fig, ax = plt.subplots(figsize=(7, 5))
ax.barh(imp_df["feature"], imp_df["importance"],
        color=imp_df["color"], edgecolor="white", alpha=0.85)
ax.set_xlabel("Mean decrease in impurity (feature importance)")
ax.set_title("Random Forest: which condition indices matter most?")

energy_patch = mpatches.Patch(color=COL_SMI, label="Energy reserves")
haem_patch   = mpatches.Patch(color=COL_PC1, label="Haematological")
cov_patch    = mpatches.Patch(color="#aaaaaa", label="Covariates")
ax.legend(handles=[energy_patch, haem_patch, cov_patch], fontsize=9)

plt.tight_layout()
fig.savefig(FIGURES / "06b_rf_importance.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: 06b_rf_importance.png")


# ── Figure 3 – Gradient Boosting partial dependence on SMI ───────────────────

gb_pipe = models["Gradient Boosting (raw indices)"]
X_raw = df[FEAT_RAW_COND].values
gb_pipe.fit(X_raw, y)
gb_model = gb_pipe.named_steps["model"]

# Manual partial dependence for SMI (index 0 in FEAT_RAW_COND)
smi_idx = FEAT_RAW_COND.index("SMIcntr")
X_mean = X_raw.copy()
smi_range = np.linspace(X_raw[:, smi_idx].min(), X_raw[:, smi_idx].max(), 200)
pd_vals = []
for val in smi_range:
    X_tmp = X_mean.copy()
    X_tmp[:, smi_idx] = val
    X_scaled = gb_pipe.named_steps["scaler"].transform(X_tmp)
    pd_vals.append(gb_model.predict(X_scaled).mean())

pd_vals_smooth = gaussian_filter1d(pd_vals, sigma=3)

fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(smi_range, pd_vals,        color=COL_SMI, linewidth=1.0, alpha=0.35, label="Raw")
ax.plot(smi_range, pd_vals_smooth, color=COL_SMI, linewidth=2.5, label="Smoothed")
ax.set_xlabel("SMI (centred)")
ax.set_ylabel("Partial dependence\n(predicted independent young)")
ax.set_title("Gradient Boosting: partial dependence on SMI")
ax.legend(fontsize=9)
plt.tight_layout()
fig.savefig(FIGURES / "06c_gb_partial_dep_smi.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: 06c_gb_partial_dep_smi.png")

print("\n[06_ml] Done.\n")
