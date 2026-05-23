# =============================================================================
# EXPERIMENT NOTEBOOK — Multilevel Logistic Regression
# Student Dropout — Pre-Tec21 vs Tec21 + School-level structure
#
# Phase 1 — Baseline:       sklearn logistic regression, no group structure
# Phase 2 — Era structure:  fixed effect for Tec21 rupture (sklearn + PyMC)
# Phase 3 — Full multilevel: school-level random intercepts (PyMC)
#
# Run with: jupyter notebook  OR  jupyter nbconvert --to notebook --execute
# Requires: pymc>=5.0, scikit-learn>=1.4, pandas, numpy, arviz, matplotlib
# =============================================================================

# %% [0] ── Imports ──────────────────────────────────────────────────────────
import json
import warnings
from pathlib import Path

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt
from scipy.special import expit  # sigmoid
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    RocCurveDisplay,
    average_precision_score,
    brier_score_loss,
    classification_report,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Paths — adjust if your layout differs
DATA_DIR = Path("../data/processed")
OUTPUT_DIR = Path("../outputs/experiments")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
rng = np.random.default_rng(SEED)

print(f"PyMC  {pm.__version__}")
print(f"ArviZ {az.__version__}")


# %% [1] ── Load data + feature manifest ─────────────────────────────────────
df = pd.read_parquet(DATA_DIR / "dataset_student_dropout_imputado_investigacion.parquet")
school_lookup = pd.read_csv(DATA_DIR / "school_code_lookup.csv")

with open(DATA_DIR / "feature_manifest.json", encoding="utf-8") as f:
    M = json.load(f)

# Convenience aliases
TARGET      = M["target"]
NUM_COMMON  = M["numeric_common"]
NUM_TEC21   = M["numeric_tec21_model"]   # was: M["numeric_tec21"]
CAT_COMMON  = M["categorical_common"]
FLAGS       = M["missing_flags"]
ERA_COL     = "era_code"                           # 0=Pre-Tec21, 1=Tec21
SCHOOL_COL  = "school_code"
N_SCHOOLS   = df[SCHOOL_COL].nunique()
N_ERAS      = 2

# Drop rows where target is missing (retention=NaN propagates to dropout=NaN)
df_model = df.dropna(subset=[TARGET]).reset_index(drop=True)

print(f"Total usable rows  : {len(df_model):,}")
print(f"Dropout rate       : {df_model[TARGET].mean():.3%}")
print(f"Schools            : {N_SCHOOLS}")
print(f"Era distribution   :")
print(df_model.groupby("era")[TARGET].agg(["count", "mean"]).rename(
    columns={"count": "n", "mean": "dropout_rate"}))


# %% [2] ── Train / test split ───────────────────────────────────────────────
# Stratify jointly by era × dropout to preserve marginal rates in both splits.
df_model["_strat"] = (
    df_model[ERA_COL].astype(str) + "_" + df_model[TARGET].astype(int).astype(str)
)

X_raw, X_test_raw, y_train, y_test = train_test_split(
    df_model.drop(columns=[TARGET]),
    df_model[TARGET].values,
    test_size=0.20,
    stratify=df_model["_strat"],
    random_state=SEED,
)
X_raw   = X_raw.reset_index(drop=True)
X_test_raw = X_test_raw.reset_index(drop=True)

# Group arrays used by PyMC (extracted BEFORE sklearn transforms)
school_train = X_raw[SCHOOL_COL].values.astype(int)
school_test  = X_test_raw[SCHOOL_COL].values.astype(int)
era_train    = X_raw[ERA_COL].values.astype(int)
era_test     = X_test_raw[ERA_COL].values.astype(int)

print(f"\nTrain : {len(y_train):,} rows  |  dropout {y_train.mean():.3%}")
print(f"Test  : {len(y_test):,}  rows  |  dropout {y_test.mean():.3%}")


# %% [3] ── Column transformer (shared across all sklearn models) ─────────────
# Strategy:
#  - StandardScaler on all numeric columns (NaN already imputed in preprocessing)
#  - OneHotEncoder on categoricals — handles 'No information' / 'Does not apply'
#    as legitimate categories (handle_unknown='infrequent_if_exist')
#  - Missing flags passed through as-is (already 0/1)

numeric_cols_m1  = NUM_COMMON                   # Model 1 & 2: common only
numeric_cols_m3  = NUM_COMMON + NUM_TEC21        # Model 3: add tec21-only filled
flag_cols        = FLAGS

def build_preprocessor(numeric_cols):
    return ColumnTransformer(
        transformers=[
            ("num",  StandardScaler(),                numeric_cols),
            ("cat",  OneHotEncoder(
                        handle_unknown="infrequent_if_exist",
                        sparse_output=False,
                        min_frequency=30,          # collapse rare categories
                    ),                             CAT_COMMON),
            ("flags", "passthrough",               flag_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

preprocessor_common = build_preprocessor(numeric_cols_m1)

# Fit once on train, reuse for Models 1 & 2
X_train_sk = preprocessor_common.fit_transform(X_raw)
X_test_sk  = preprocessor_common.transform(X_test_raw)

feature_names = preprocessor_common.get_feature_names_out()
print(f"sklearn feature matrix: {X_train_sk.shape[1]} columns")
print(f"  numeric  : {len(numeric_cols_m1)}")
print(f"  one-hot  : {X_train_sk.shape[1] - len(numeric_cols_m1) - len(flag_cols)}")
print(f"  flags    : {len(flag_cols)}")


# %% [4] ── Evaluation helper ────────────────────────────────────────────────
def evaluate(name, y_true, y_prob, threshold=0.5):
    y_pred = (y_prob >= threshold).astype(int)
    auc    = roc_auc_score(y_true, y_prob)
    ap     = average_precision_score(y_true, y_prob)
    brier  = brier_score_loss(y_true, y_prob)
    print(f"\n{'─'*50}")
    print(f"  {name}")
    print(f"{'─'*50}")
    print(f"  ROC-AUC  : {auc:.4f}")
    print(f"  Avg Prec : {ap:.4f}")
    print(f"  Brier    : {brier:.4f}")
    print(classification_report(y_true, y_pred, target_names=["retained", "dropout"]))
    return {"model": name, "roc_auc": auc, "avg_prec": ap, "brier": brier}

results = []   # accumulates dicts for final comparison table


# ─────────────────────────────────────────────────────────────────────────────
# ██████████████  PHASE 1 — BASELINE LOGISTIC REGRESSION  ████████████████████
# ─────────────────────────────────────────────────────────────────────────────

# %% [5] ── Model 1: Baseline sklearn logistic regression ────────────────────
# No group structure. L2-regularized, saga solver for large datasets.
# C=1 corresponds to λ=1 (moderate regularization); tune via CV if needed.

lr_baseline = LogisticRegression(
    C=1.0,
    solver="saga",
    penalty="l2",
    max_iter=2000,
    class_weight="balanced",   # compensates class imbalance
    random_state=SEED,
    n_jobs=-1,
)
lr_baseline.fit(X_train_sk, y_train)

prob_train_m1 = lr_baseline.predict_proba(X_train_sk)[:, 1]
prob_test_m1  = lr_baseline.predict_proba(X_test_sk)[:, 1]

results.append(evaluate("M1 – Baseline LR (no groups)", y_test, prob_test_m1))

# Top absolute coefficients
coef_df = pd.DataFrame({
    "feature": feature_names,
    "coef":    lr_baseline.coef_[0],
}).sort_values("coef", key=abs, ascending=False)

print("\nTop 15 features by |coef|:")
display(coef_df.head(15))


# ─────────────────────────────────────────────────────────────────────────────
# ██████████████  PHASE 2 — ERA FIXED EFFECT  █████████████████████████████████
# ─────────────────────────────────────────────────────────────────────────────

# %% [6] ── Model 2a: sklearn LR + era dummy ──────────────────────────────────
# Simplest way to add the temporal rupture: append era_code as an extra feature.
# This adds a global intercept shift for Tec21 students.

X_train_m2 = np.column_stack([X_train_sk, era_train])
X_test_m2  = np.column_stack([X_test_sk,  era_test])

lr_era = LogisticRegression(
    C=1.0,
    solver="saga",
    penalty="l2",
    max_iter=2000,
    class_weight="balanced",
    random_state=SEED,
    n_jobs=-1,
)
lr_era.fit(X_train_m2, y_train)

prob_test_m2a = lr_era.predict_proba(X_test_m2)[:, 1]
results.append(evaluate("M2a – LR + era fixed effect (sklearn)", y_test, prob_test_m2a))

era_coef = lr_era.coef_[0, -1]
print(f"\nCoefficient for era_code (Tec21 vs Pre-Tec21): {era_coef:.4f}")
print(f"  → Tec21 log-odds shift: {era_coef:+.4f}  (OR = {np.exp(era_coef):.3f})")


# %% [6b] ── Model 2b: PyMC — separate intercepts per era (partial pooling) ───
# With only 2 groups, partial pooling ≈ shrinkage toward shared mean.
# This is the foundation for the full multilevel model.
#
# Model:
#   logit(p_i) = α[era_i] + X_i @ β
#   α[k] ~ Normal(μ_α, σ_α)    k ∈ {0=Pre-Tec21, 1=Tec21}
#   μ_α  ~ Normal(0, 1.5)
#   σ_α  ~ HalfNormal(1)
#   β    ~ Normal(0, 1)         (weakly informative on standardised features)

coords_era = {
    "era":     ["Pre-Tec21", "Tec21"],
    "feature": list(feature_names),
}

X_train_pt = X_train_sk.astype("float32")
X_test_pt  = X_test_sk.astype("float32")

with pm.Model(coords=coords_era) as era_model:
    # Data containers — allows out-of-sample prediction without re-compiling
    X_data    = pm.Data("X",        X_train_pt,  dims=("obs", "feature"))
    era_data  = pm.Data("era_idx",  era_train,   dims="obs")

    # Hyperpriors for era intercepts
    mu_alpha  = pm.Normal("mu_alpha", 0.0, 1.5)
    sigma_alpha = pm.HalfNormal("sigma_alpha", 1.0)

    # Era-level intercepts (non-centered)
    z_era   = pm.Normal("z_era", 0.0, 1.0, dims="era")
    alpha   = pm.Deterministic("alpha", mu_alpha + sigma_alpha * z_era, dims="era")

    # Feature coefficients
    beta    = pm.Normal("beta", 0.0, 1.0, dims="feature")

    # Linear predictor
    logit_p = alpha[era_data] + pm.math.dot(X_data, beta)

    # Likelihood
    y_obs   = pm.Bernoulli("y_obs", logit_p=logit_p, observed=y_train)

# ── Inference: start with MAP for a fast sanity check ───────────────────────
with era_model:
    map_era = pm.find_MAP(progressbar=True)
    logit_p_test_era = (
        map_era["alpha"][era_test]
        + X_test_pt @ map_era["beta"]
    )
    prob_test_m2b_map = expit(logit_p_test_era)

results.append(evaluate("M2b – PyMC era partial pooling (MAP)", y_test, prob_test_m2b_map))

# ── Full NUTS sampling (comment out if dataset subset not used) ──────────────
# NOTE: 143k rows × full NUTS is expensive (~hours on CPU).
# Options:
#   A) Sample a balanced subset for MCMC, use MAP for production scoring.
#   B) Use ADVI (fast variational approximation).
#   C) Use GPU / JAX backend.
# Here we show ADVI as the practical default.
if RUN_NUTS:
    
with era_model:
    approx_era = pm.fit(
        n=30_000,
        method="advi",
        random_seed=SEED,
        progressbar=True,
    )
    trace_era = approx_era.sample(2000, random_seed=SEED)

# Posterior predictive on test set
with era_model:
    pm.set_data({"X": X_test_pt, "era_idx": era_test})
    ppc_era = pm.sample_posterior_predictive(
        trace_era, var_names=["y_obs"], random_seed=SEED, progressbar=False
    )

prob_test_m2b = ppc_era.posterior_predictive["y_obs"].mean(dim=["chain", "draw"]).values
results.append(evaluate("M2b – PyMC era partial pooling (ADVI)", y_test, prob_test_m2b))

print("\nEra intercepts (ADVI posterior mean):")
alpha_post = trace_era.posterior["alpha"].mean(dim=["chain", "draw"]).values
for k, name in enumerate(["Pre-Tec21", "Tec21"]):
    print(f"  α[{name}] = {alpha_post[k]:.3f}  (OR = {np.exp(alpha_post[k]):.3f})")


# ─────────────────────────────────────────────────────────────────────────────
# ██████████████  PHASE 3 — FULL MULTILEVEL MODEL  ████████████████████████████
# ─────────────────────────────────────────────────────────────────────────────

# %% [7] ── Preprocessor for Model 3: adds Tec21-only filled vars ─────────────
preprocessor_m3  = build_preprocessor(NUM_COMMON + NUM_TEC21)
X_train_m3 = preprocessor_m3.fit_transform(X_raw).astype("float32")
X_test_m3  = preprocessor_m3.transform(X_test_raw).astype("float32")
feature_names_m3 = preprocessor_m3.get_feature_names_out()

print(f"Model 3 feature matrix: {X_train_m3.shape[1]} columns")
print(f"  common numeric   : {len(NUM_COMMON)}")
print(f"  tec21-only model : {len(NUM_TEC21)}")

# For the Tec21-only _filled columns: multiply by era_code so Pre-Tec21 rows
# contribute exactly 0 even though they were filled with -1.
# This creates an implicit interaction: β_tec21_feat × era_code × value.
tec21_col_indices = [
    i for i, fn in enumerate(feature_names_m3)
    if any(tc.replace("_filled", "") in fn for tc in
           ["average.first.period", "failed.subject.first.period", "dropped.subject.first.period"])
]
X_train_m3_raw[:, tec21_col_indices] *= era_train[:, None]
X_test_m3_raw[:,  tec21_col_indices] *= era_test[:,  None]

X_train_m3 = X_train_m3_raw.astype("float32")
X_test_m3  = X_test_m3_raw.astype("float32")

print(f"Model 3 feature matrix: {X_train_m3.shape[1]} columns")


# %% [8] ── Model 3: PyMC — school random intercepts + era fixed effect ────────
#
# Generative structure:
#
#   Level 1 (student):
#     logit(p_i) = α[school_i] + β_era × era_i + X_i @ β
#
#   Level 2 (school):
#     α[j]    ~ Normal(μ_α, σ_α)    — random intercepts, partial pooling
#
#   Hyperpriors:
#     μ_α     ~ Normal(0, 1.5)      — grand mean log-odds
#     σ_α     ~ HalfNormal(1)       — between-school SD
#     β_era   ~ Normal(0, 1)        — era fixed effect
#     β       ~ Normal(0, 1)        — student-level predictors
#
# Non-centered parameterization for α avoids funnel geometry in NUTS/ADVI.

n_features_m3 = X_train_m3.shape[1]

coords_full = {
    "school":   school_lookup["school"].tolist(),
    "feature":  list(feature_names_m3),
}

with pm.Model(coords=coords_full) as multilevel_model:
    # ── Data containers ──────────────────────────────────────────────────────
    X_data_m3    = pm.Data("X",          X_train_m3,    dims=("obs", "feature"))
    era_data_m3  = pm.Data("era_idx",    era_train,      dims="obs")
    school_data  = pm.Data("school_idx", school_train,   dims="obs")

    # ── Level-2 hyperpriors ──────────────────────────────────────────────────
    mu_alpha    = pm.Normal("mu_alpha",   0.0, 1.5)
    sigma_alpha = pm.HalfNormal("sigma_alpha", 1.0)

    # ── School random intercepts (non-centered) ──────────────────────────────
    z_school    = pm.Normal("z_school", 0.0, 1.0, dims="school")
    alpha_school = pm.Deterministic(
        "alpha_school",
        mu_alpha + sigma_alpha * z_school,
        dims="school",
    )

    # ── Era fixed effect ─────────────────────────────────────────────────────
    beta_era    = pm.Normal("beta_era", 0.0, 1.0)

    # ── Student-level feature coefficients ───────────────────────────────────
    beta        = pm.Normal("beta", 0.0, 1.0, dims="feature")

    # ── Linear predictor ─────────────────────────────────────────────────────
    logit_p = (
        alpha_school[school_data]
        + beta_era * era_data_m3
        + pm.math.dot(X_data_m3, beta)
    )

    # ── Likelihood ───────────────────────────────────────────────────────────
    y_obs = pm.Bernoulli("y_obs", logit_p=logit_p, observed=y_train)


# %% [8b] ── Fit Model 3 (MAP → ADVI → optional NUTS) ────────────────────────

# ── Step 1: MAP (seconds, good sanity check) ─────────────────────────────────
with multilevel_model:
    map_full = pm.find_MAP(progressbar=True)

logit_p_test_map = (
    map_full["alpha_school"][school_test]
    + map_full["beta_era"] * era_test
    + X_test_m3 @ map_full["beta"]
)
prob_test_m3_map = expit(logit_p_test_map)
results.append(evaluate("M3 – Multilevel school+era (MAP)", y_test, prob_test_m3_map))

# ── Step 2: ADVI (~minutes) ──────────────────────────────────────────────────
with multilevel_model:
    approx_full = pm.fit(
        n=50_000,
        method="advi",
        random_seed=SEED,
        progressbar=True,
    )
    trace_full = approx_full.sample(2000, random_seed=SEED)

# ── Step 3: Posterior predictive on test set ─────────────────────────────────
with multilevel_model:
    pm.set_data({
        "X":          X_test_m3,
        "era_idx":    era_test,
        "school_idx": school_test,
    })
    ppc_full = pm.sample_posterior_predictive(
        trace_full, var_names=["y_obs"], random_seed=SEED, progressbar=False
    )

prob_test_m3 = ppc_full.posterior_predictive["y_obs"].mean(dim=["chain", "draw"]).values
results.append(evaluate("M3 – Multilevel school+era (ADVI)", y_test, prob_test_m3))


# %% [9] ── School-level variance decomposition ───────────────────────────────
# How much of the log-odds variance is explained by school vs individual?

alpha_samples = trace_full.posterior["alpha_school"]      # (chain, draw, school)
sigma_alpha_samples = trace_full.posterior["sigma_alpha"] # (chain, draw)

sigma_alpha_mean = float(sigma_alpha_samples.mean())
sigma_alpha_hdi  = az.hdi(trace_full, var_names=["sigma_alpha"])["sigma_alpha"].values

print("\n── Between-school variance ─────────────────────────────────────────")
print(f"  σ_α posterior mean : {sigma_alpha_mean:.3f}")
print(f"  σ_α 94% HDI        : [{sigma_alpha_hdi[0]:.3f}, {sigma_alpha_hdi[1]:.3f}]")

# School-level caterpillar plot (sorted by posterior mean)
alpha_mean = alpha_samples.mean(dim=["chain", "draw"]).values
alpha_hdi  = az.hdi(trace_full, var_names=["alpha_school"])["alpha_school"].values

school_effect_df = pd.DataFrame({
    "school":      coords_full["school"],
    "alpha_mean":  alpha_mean,
    "hdi_low":     alpha_hdi[:, 0],
    "hdi_high":    alpha_hdi[:, 1],
}).sort_values("alpha_mean")

fig, ax = plt.subplots(figsize=(8, max(4, len(coords_full["school"]) * 0.35)))
for i, row in enumerate(school_effect_df.itertuples()):
    ax.plot([row.hdi_low, row.hdi_high], [i, i], color="steelblue", lw=1.5)
    ax.plot(row.alpha_mean, i, "o", color="steelblue", ms=4)
ax.axvline(0, color="grey", lw=0.8, ls="--")
ax.set_yticks(range(len(school_effect_df)))
ax.set_yticklabels(school_effect_df["school"].tolist(), fontsize=8)
ax.set_xlabel("School random intercept α_j  (log-odds scale)")
ax.set_title("School-level random intercepts — 94% HDI\n(positive = higher dropout tendency, net of covariates)")
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "school_random_intercepts.png", dpi=150)
plt.show()
print(f"\nCaterpillar plot saved → {OUTPUT_DIR / 'school_random_intercepts.png'}")

# Era fixed effect interpretation
beta_era_post = trace_full.posterior["beta_era"]
beta_era_mean = float(beta_era_post.mean())
beta_era_hdi  = az.hdi(trace_full, var_names=["beta_era"])["beta_era"].values

print(f"\n── Era (Tec21) fixed effect ────────────────────────────────────────")
print(f"  β_era posterior mean : {beta_era_mean:.3f}")
print(f"  β_era 94% HDI        : [{beta_era_hdi[0]:.3f}, {beta_era_hdi[1]:.3f}]")
print(f"  Odds ratio           : {np.exp(beta_era_mean):.3f}")
print(f"  (positive = Tec21 associated with higher dropout on log-odds scale)")


# %% [10] ── Model comparison table ──────────────────────────────────────────
results_df = pd.DataFrame(results)
print("\n" + "="*60)
print("  MODEL COMPARISON — Test set")
print("="*60)
display(results_df.set_index("model").sort_values("roc_auc", ascending=False))

results_df.to_csv(OUTPUT_DIR / "model_comparison.csv", index=False)

# ROC curves
fig, ax = plt.subplots(figsize=(7, 5))
model_probs = [
    ("M1 – Baseline",         prob_test_m1),
    ("M2a – Era LR",           prob_test_m2a),
    ("M2b – Era PyMC (ADVI)",  prob_test_m2b),
    ("M3 – Multilevel (ADVI)", prob_test_m3),
]
for name, prob in model_probs:
    RocCurveDisplay.from_predictions(
        y_test, prob, name=name, ax=ax, plot_chance_level=(name == model_probs[-1][0])
    )
ax.set_title("ROC curves — test set")
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "roc_curves.png", dpi=150)
plt.show()
print(f"ROC plot saved → {OUTPUT_DIR / 'roc_curves.png'}")


# %% [11] ── OPTIONAL: Full NUTS on a balanced subsample ─────────────────────
# Use this block when you want reliable posterior uncertainty estimates
# but cannot afford NUTS on all 143k rows.
# A 20k balanced sample preserves class and school distributions.

if RUN_NUTS:
    sub_idx = (
        pd.DataFrame({"school": school_train, "y": y_train, "idx": np.arange(len(y_train))})
        .groupby(["school", "y"], group_keys=False)
        .apply(lambda g: g.sample(min(len(g), 200), random_state=SEED))  # ≤200/cell
        ["idx"].values
    )
    X_sub     = X_train_m3[sub_idx]
    era_sub   = era_train[sub_idx]
    school_sub = school_train[sub_idx]
    y_sub     = y_train[sub_idx]

    print(f"NUTS subsample: {len(y_sub):,} rows  |  dropout {y_sub.mean():.3%}")

    with pm.Model(coords=coords_full) as nuts_model:
        X_data_n    = pm.Data("X",          X_sub.astype("float32"), dims=("obs", "feature"))
        era_data_n  = pm.Data("era_idx",    era_sub,                  dims="obs")
        school_data_n = pm.Data("school_idx", school_sub,             dims="obs")

        mu_alpha    = pm.Normal("mu_alpha",    0.0, 1.5)
        sigma_alpha = pm.HalfNormal("sigma_alpha", 1.0)
        z_school    = pm.Normal("z_school",    0.0, 1.0, dims="school")
        alpha_school = pm.Deterministic(
            "alpha_school", mu_alpha + sigma_alpha * z_school, dims="school"
        )
        beta_era    = pm.Normal("beta_era",  0.0, 1.0)
        beta        = pm.Normal("beta",      0.0, 1.0, dims="feature")

        logit_p = (
            alpha_school[school_data_n]
            + beta_era * era_data_n
            + pm.math.dot(X_data_n, beta)
        )
        pm.Bernoulli("y_obs", logit_p=logit_p, observed=y_sub)

    with nuts_model:
        trace_nuts = pm.sample(
            draws=1000, tune=1000, chains=4,
            target_accept=0.9,
            random_seed=SEED,
            progressbar=True,
        )

    print(az.summary(trace_nuts, var_names=["mu_alpha", "sigma_alpha", "beta_era"],
                     hdi_prob=0.94))

    trace_nuts.to_netcdf(OUTPUT_DIR / "trace_nuts_subsample.nc")
    print(f"\nNUTS trace saved → {OUTPUT_DIR / 'trace_nuts_subsample.nc'}")


# %% [12] ── Save traces for later analysis ──────────────────────────────────
trace_era.to_netcdf( OUTPUT_DIR / "trace_era_advi.nc")
trace_full.to_netcdf(OUTPUT_DIR / "trace_multilevel_advi.nc")
print("Traces saved.")
print("\nDone. Outputs in:", OUTPUT_DIR.resolve())
