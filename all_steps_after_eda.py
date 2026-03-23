# ═══════════════════════════════════════════════════════
#  INDOSTAR — ALL STEPS AFTER EDA
#  Short cells, easy to type one by one
# ═══════════════════════════════════════════════════════


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  STEP 3 — BAD RATE BINNING
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# BINNING CELL 1 — Define binning function
def bad_rate_bin(df, col, n=10):
    tmp = df[[col, "TARGET_1to1PLUS"]].dropna().copy()
    try:
        tmp["bin"] = pd.qcut(tmp[col], q=n, duplicates="drop")
    except:
        tmp["bin"] = pd.cut(tmp[col], bins=n, duplicates="drop")
    out = (
        tmp.groupby("bin", observed=True)["TARGET_1to1PLUS"]
        .agg(Total="count", Bad="sum")
        .assign(
            Good     = lambda x: x["Total"] - x["Bad"],
            Bad_Rate = lambda x: (x["Bad"] / x["Total"] * 100).round(2)
        )
        .reset_index()
    )
    return out

print("Binning function ready")


# BINNING CELL 2 — Run binning on all selected vars
selected_vars = [
    "DPD", "OVER_DUE_POS", "DPD_Transitions_3M",
    "TrendDPD_3M", "MaxDPD_12M", "MaxDPD_24M",
    "Latest_DPD", "DPD_Std_6M", "DPD_Std_12M",
    "CountInBkt_1Plus_6M", "CountInBkt_1Plus_12M",
    "WeightedAvgBkt_6M", "BounceRate_6M",
    "BounceRate_12M", "ChronicBouncer",
    "WeightedBounceScore", "coll_count_L6M",
    "coll_prop_L12M", "coll_rw_score",
    "EarlyWarningFlag", "DPD_Slope_6M",
    "MOB", "pct_paid", "IRR",
    "APPLICANT_CIBIL_SCORE"
]

present_vars = [v for v in selected_vars if v in df.columns]
print("Variables found:", len(present_vars))
print("Variables missing:", [v for v in selected_vars if v not in df.columns])


# BINNING CELL 3 — Print bin tables for each variable
for col in present_vars:
    try:
        s = bad_rate_bin(df, col)
        print("\n" + "="*50)
        print(f"Variable: {col}")
        print("="*50)
        print(s[["bin","Total","Bad","Bad_Rate"]].to_string(index=False))
    except Exception as e:
        print(f"Skipped {col}: {e}")


# BINNING CELL 4 — Plot bad rate chart for one variable
# Change col= to any variable you want to visualise
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

def plot_bin(df, col):
    s = bad_rate_bin(df, col)
    fig, ax1 = plt.subplots(figsize=(10, 4))
    ax2 = ax1.twinx()
    ax1.bar(s["bin"].astype(str), s["Total"],
            color="#B5D5C5", label="Volume")
    ax2.plot(s["bin"].astype(str), s["Bad_Rate"],
             "o-", color="#E63946", lw=2, ms=5)
    ax2.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax1.set_ylabel("Count")
    ax2.set_ylabel("Bad Rate %")
    plt.title(f"Bad Rate by Bin — {col}", fontweight="bold")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(f"bin_{col}.png", dpi=150)
    plt.show()
    print(f"Saved: bin_{col}.png")

# Plot for key variables
for col in ["DPD", "MaxDPD_12M", "BounceRate_12M",
            "DPD_Std_6M", "coll_rw_score"]:
    if col in df.columns:
        plot_bin(df, col)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  STEP 4 — WOE AND IV
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# IV CELL 1 — Define WOE/IV function
def compute_iv(df, col, n=10):
    tmp = df[[col, "TARGET_1to1PLUS"]].dropna().copy()
    total_bad  = tmp["TARGET_1to1PLUS"].sum()
    total_good = len(tmp) - total_bad
    if total_bad == 0 or total_good == 0:
        return 0.0
    try:
        tmp["bin"] = pd.qcut(tmp[col], q=n, duplicates="drop")
    except:
        tmp["bin"] = pd.cut(tmp[col], bins=n, duplicates="drop")
    g = (
        tmp.groupby("bin", observed=True)["TARGET_1to1PLUS"]
        .agg(bad="sum", total="count")
        .assign(good=lambda x: x["total"] - x["bad"])
    )
    g["dist_bad"]  = (g["bad"]  + 0.5) / (total_bad  + 0.5)
    g["dist_good"] = (g["good"] + 0.5) / (total_good + 0.5)
    g["woe"] = np.log(g["dist_good"] / g["dist_bad"])
    g["iv"]  = (g["dist_good"] - g["dist_bad"]) * g["woe"]
    return round(g["iv"].sum(), 4)

print("IV function ready")


# IV CELL 2 — Compute IV for all selected variables
iv_results = []

for col in present_vars:
    try:
        iv = compute_iv(df, col)
        if iv > 0.5:
            strength = "Suspicious"
        elif iv > 0.3:
            strength = "Strong"
        elif iv > 0.1:
            strength = "Medium"
        elif iv > 0.02:
            strength = "Weak"
        else:
            strength = "Useless"
        iv_results.append({
            "Variable" : col,
            "IV"       : iv,
            "Strength" : strength
        })
    except Exception as e:
        print(f"Skipped {col}: {e}")

iv_table = pd.DataFrame(iv_results).sort_values(
    "IV", ascending=False
).reset_index(drop=True)

print(iv_table.to_string(index=False))


# IV CELL 3 — IV bar chart
plt.figure(figsize=(10, 8))

colors = [
    "#E63946" if v > 0.3
    else "#457B9D" if v > 0.1
    else "#8B949E"
    for v in iv_table["IV"]
]

plt.barh(
    iv_table["Variable"][::-1],
    iv_table["IV"][::-1],
    color=colors[::-1]
)
plt.axvline(0.1, color="orange", ls="--",
            lw=1.5, label="0.1 = Medium")
plt.axvline(0.3, color="red", ls="--",
            lw=1.5, label="0.3 = Strong")
plt.xlabel("Information Value (IV)")
plt.title("IV — All Selected Variables", fontweight="bold")
plt.legend()
plt.tight_layout()
plt.savefig("iv_chart.png", dpi=150)
plt.show()
print("Saved: iv_chart.png")


# IV CELL 4 — Final confirmed variables after IV filter
confirmed_vars = iv_table[
    (iv_table["IV"] >= 0.02) &
    (iv_table["IV"] <= 0.5)
]["Variable"].tolist()

print("Variables kept (IV 0.02 to 0.5):", len(confirmed_vars))
print(confirmed_vars)

dropped_vars = iv_table[
    (iv_table["IV"] < 0.02) |
    (iv_table["IV"] > 0.5)
]["Variable"].tolist()

print("\nVariables dropped:", dropped_vars)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  STEP 5 — FEATURE SELECTION
#  (Correlation check — remove similar variables)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# FEATURE CELL 1 — Correlation heatmap
import seaborn as sns

corr = df[confirmed_vars].corr().round(2)

plt.figure(figsize=(14, 10))
sns.heatmap(
    corr,
    annot=True,
    fmt=".2f",
    cmap="RdBu_r",
    center=0,
    linewidths=0.5,
    annot_kws={"size": 7}
)
plt.title("Correlation Matrix — Selected Variables",
          fontweight="bold")
plt.tight_layout()
plt.savefig("correlation_matrix.png", dpi=150)
plt.show()
print("Saved: correlation_matrix.png")


# FEATURE CELL 2 — Remove highly correlated variables
def remove_correlated(df, features, threshold=0.85):
    corr = df[features].corr().abs()
    upper = corr.where(
        np.triu(np.ones(corr.shape), k=1).astype(bool)
    )
    to_drop = [
        col for col in upper.columns
        if any(upper[col] > threshold)
    ]
    kept = [c for c in features if c not in to_drop]
    print(f"Threshold : {threshold}")
    print(f"Dropped   : {to_drop}")
    print(f"Kept      : {len(kept)} variables")
    return kept

final_vars = remove_correlated(df, confirmed_vars)
print("\nFinal variables for model:")
print(final_vars)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  STEP 6 — TRAIN / TEST SPLIT
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# SPLIT CELL 1 — Split data by month
TRAIN_MONTHS = [
    "JUL24","AUG24","SEP24","OCT24","NOV24","DEC24",
    "JAN25","FEB25","MAR25","APR25","MAY25","JUN25",
    "JUL25","AUG25","SEP25","OCT25"
]
TEST_MONTHS = ["NOV25","DEC25","JAN26"]

train_df = df[df["MONTH"].isin(TRAIN_MONTHS)].copy()
test_df  = df[df["MONTH"].isin(TEST_MONTHS)].copy()

X_train = train_df[final_vars].fillna(-999)
y_train = train_df["TARGET_1to1PLUS"]
X_test  = test_df[final_vars].fillna(-999)
y_test  = test_df["TARGET_1to1PLUS"]

print("TRAIN:", X_train.shape,
      "| Bad Rate:", round(y_train.mean()*100, 2), "%")
print("TEST :", X_test.shape,
      "| Bad Rate:", round(y_test.mean()*100, 2), "%")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  STEP 7 — LOGISTIC REGRESSION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# LR CELL 1 — Train model
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, roc_curve

scaler  = StandardScaler()
X_tr_s  = scaler.fit_transform(X_train)
X_te_s  = scaler.transform(X_test)

lr = LogisticRegression(
    max_iter     = 1000,
    class_weight = "balanced",
    C            = 0.1,
    solver       = "lbfgs",
    random_state = 42
)
lr.fit(X_tr_s, y_train)
print("Logistic Regression trained")


# LR CELL 2 — Evaluate LR
lr_prob_train = lr.predict_proba(X_tr_s)[:, 1]
lr_prob_test  = lr.predict_proba(X_te_s)[:, 1]

lr_auc  = roc_auc_score(y_test, lr_prob_test)
lr_gini = 2 * lr_auc - 1
fpr, tpr, _ = roc_curve(y_test, lr_prob_test)
lr_ks   = round(float(max(tpr - fpr)), 4)

print("── Logistic Regression Results ──")
print(f"AUC  : {lr_auc:.4f}")
print(f"Gini : {lr_gini:.4f}")
print(f"KS   : {lr_ks:.4f}")


# LR CELL 3 — Coefficient table (which variables matter most)
coef_df = pd.DataFrame({
    "Variable"   : final_vars,
    "Coefficient": lr.coef_[0]
}).sort_values("Coefficient", key=abs, ascending=False)

print("Top 10 most important variables (LR):")
print(coef_df.head(10).to_string(index=False))


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  STEP 8 — XGBOOST
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# XGB CELL 1 — Train XGBoost
import xgboost as xgb

spw = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
print(f"Scale pos weight: {spw:.2f}")

xgb_model = xgb.XGBClassifier(
    n_estimators          = 300,
    max_depth             = 4,
    learning_rate         = 0.05,
    subsample             = 0.8,
    colsample_bytree      = 0.8,
    scale_pos_weight      = spw,
    eval_metric           = "auc",
    use_label_encoder     = False,
    early_stopping_rounds = 20,
    random_state          = 42,
    verbosity             = 1
)
xgb_model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    verbose=50
)
print("XGBoost trained")


# XGB CELL 2 — Evaluate XGBoost
xgb_prob_train = xgb_model.predict_proba(X_train)[:, 1]
xgb_prob_test  = xgb_model.predict_proba(X_test)[:, 1]

xgb_auc  = roc_auc_score(y_test, xgb_prob_test)
xgb_gini = 2 * xgb_auc - 1
fpr, tpr, _ = roc_curve(y_test, xgb_prob_test)
xgb_ks  = round(float(max(tpr - fpr)), 4)

print("── XGBoost Results ──")
print(f"AUC  : {xgb_auc:.4f}")
print(f"Gini : {xgb_gini:.4f}")
print(f"KS   : {xgb_ks:.4f}")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  STEP 9 — EVALUATION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# EVAL CELL 1 — Model comparison table
comparison = pd.DataFrame({
    "Model": ["Logistic Regression", "XGBoost"],
    "AUC"  : [round(lr_auc, 4),  round(xgb_auc, 4)],
    "Gini" : [round(lr_gini, 4), round(xgb_gini, 4)],
    "KS"   : [lr_ks, xgb_ks]
})
print(comparison.to_string(index=False))
print()
print("Benchmark targets:")
print("  AUC  > 0.70  Good | > 0.75  Great")
print("  Gini > 0.40  Good | > 0.50  Great")
print("  KS   > 0.30  Good | > 0.40  Great")


# EVAL CELL 2 — ROC curve chart
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for name, prob, auc, ks in [
    ("Logistic Regression", lr_prob_test,  lr_auc,  lr_ks),
    ("XGBoost",             xgb_prob_test, xgb_auc, xgb_ks)
]:
    fpr, tpr, _ = roc_curve(y_test, prob)
    axes[0].plot(fpr, tpr, lw=2,
                 label=f"{name} AUC={auc:.3f} KS={ks:.3f}")

axes[0].plot([0,1],[0,1],"k--", lw=1)
axes[0].set_xlabel("False Positive Rate")
axes[0].set_ylabel("True Positive Rate")
axes[0].set_title("ROC Curve")
axes[0].legend(fontsize=9)

# KS plot for XGBoost
t = pd.DataFrame({
    "y": y_test.values,
    "p": xgb_prob_test
}).sort_values("p")

t["cum_bad"]  = t["y"].cumsum() / t["y"].sum()
t["cum_good"] = (1-t["y"]).cumsum() / (1-t["y"]).sum()

axes[1].plot(range(len(t)), t["cum_bad"],
             color="#E63946", label="Cum Bad")
axes[1].plot(range(len(t)), t["cum_good"],
             color="#457B9D", label="Cum Good")
axes[1].fill_between(range(len(t)),
                     t["cum_bad"], t["cum_good"],
                     alpha=0.12, color="green")
axes[1].set_title(f"KS Plot — XGBoost (KS={xgb_ks:.3f})")
axes[1].legend()

plt.suptitle("Model Evaluation — TARGET_1to1PLUS",
             fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("model_evaluation.png", dpi=150)
plt.show()
print("Saved: model_evaluation.png")


# EVAL CELL 3 — PSI (model stability check)
def compute_psi(expected, actual, n=10):
    bins = np.percentile(expected,
                         np.linspace(0, 100, n+1))
    bins[0]  -= 1e-6
    bins[-1] += 1e-6
    e = np.clip(
        np.histogram(expected, bins)[0] / len(expected),
        1e-6, None
    )
    a = np.clip(
        np.histogram(actual, bins)[0] / len(actual),
        1e-6, None
    )
    psi = float(np.sum((a - e) * np.log(a / e)))
    if psi < 0.1:
        status = "Stable"
    elif psi < 0.25:
        status = "Monitor"
    else:
        status = "Drift — retrain needed"
    print(f"PSI = {round(psi,4)} → {status}")
    return psi

print("── PSI Check ──")
print("Logistic Regression:")
compute_psi(lr_prob_train, lr_prob_test)
print("XGBoost:")
compute_psi(xgb_prob_train, xgb_prob_test)


# EVAL CELL 4 — Per month evaluation
print("── Per Month Evaluation ──")
rows = []
for month in TEST_MONTHS:
    mask = test_df["MONTH"].values == month
    sub  = test_df[mask]
    prob = xgb_prob_test[mask]
    if len(sub) < 10:
        continue
    fpr, tpr, _ = roc_curve(sub["TARGET_1to1PLUS"], prob)
    ks  = round(float(max(tpr - fpr)), 4)
    auc = round(roc_auc_score(
        sub["TARGET_1to1PLUS"], prob), 4)
    rows.append({
        "Month"   : month,
        "N"       : len(sub),
        "Bad_Rate": round(
            sub["TARGET_1to1PLUS"].mean()*100, 2),
        "AUC"     : auc,
        "KS"      : ks
    })
print(pd.DataFrame(rows).to_string(index=False))


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  STEP 10 — SCORE DECILE ANALYSIS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# DECILE CELL 1 — Build decile table
test_out = test_df[["MONTH","TARGET_1to1PLUS"]].copy()
test_out["Roll_Prob"] = xgb_prob_test

test_out["Decile"] = 10 - pd.qcut(
    test_out["Roll_Prob"],
    q=10,
    labels=False,
    duplicates="drop"
)

overall_br = test_out["TARGET_1to1PLUS"].mean() * 100

deciles = (
    test_out.groupby("Decile")
    .agg(
        N        = ("TARGET_1to1PLUS", "count"),
        Bad      = ("TARGET_1to1PLUS", "sum")
    )
    .assign(
        Bad_Rate = lambda x: (
            x["Bad"] / x["N"] * 100).round(2),
        Lift     = lambda x: (
            x["Bad"] / x["N"] * 100 / overall_br
        ).round(2),
        Pct_Bad  = lambda x: (
            x["Bad"] / x["Bad"].sum() * 100).round(2)
    )
)
deciles["Cum_Bad_Pct"] = deciles["Pct_Bad"].cumsum().round(2)

print(f"Overall Bad Rate: {overall_br:.2f}%\n")
print(deciles.to_string())


# DECILE CELL 2 — Decile chart
fig, ax1 = plt.subplots(figsize=(11, 5))
ax2 = ax1.twinx()

ax1.bar(deciles.index.astype(str),
        deciles["N"], color="#B5D5C5", label="Volume")
ax2.plot(deciles.index.astype(str),
         deciles["Bad_Rate"], "o-",
         color="#E63946", lw=2.5, ms=6)
ax2.axhline(overall_br, color="#999", ls="--",
            lw=1.5, label=f"Overall {overall_br:.1f}%")

ax2.yaxis.set_major_formatter(mtick.PercentFormatter())
ax1.set_xlabel("Decile (1 = Highest Risk)")
ax1.set_ylabel("Volume")
ax2.set_ylabel("Bad Rate %")
plt.title("Score Decile Analysis — TARGET_1to1PLUS",
          fontweight="bold")
fig.legend(loc="upper right")
plt.tight_layout()
plt.savefig("score_deciles.png", dpi=150)
plt.show()
print("Saved: score_deciles.png")


# DECILE CELL 3 — SHAP feature importance
import shap

explainer   = shap.TreeExplainer(xgb_model)
shap_values = explainer.shap_values(X_test)

plt.figure(figsize=(10, 7))
shap.summary_plot(
    shap_values, X_test,
    feature_names=final_vars,
    plot_type="bar",
    show=False
)
plt.title("Feature Importance — SHAP", fontweight="bold")
plt.tight_layout()
plt.savefig("shap_importance.png", dpi=150)
plt.show()
print("Saved: shap_importance.png")


# DECILE CELL 4 — Final summary printout
print("=" * 50)
print("  PIPELINE COMPLETE — FINAL SUMMARY")
print("=" * 50)
print(f"\nData      : {df.shape[0]:,} rows")
print(f"Features  : {len(final_vars)} variables")
print(f"Train     : {len(train_df):,} rows")
print(f"Test      : {len(test_df):,} rows")
print(f"\nLogistic Regression:")
print(f"  AUC={lr_auc:.4f} Gini={lr_gini:.4f} KS={lr_ks:.4f}")
print(f"\nXGBoost:")
print(f"  AUC={xgb_auc:.4f} Gini={xgb_gini:.4f} KS={xgb_ks:.4f}")
print(f"\nTop 3 risky deciles capture:")
print(f"  {deciles.loc[1:3,'Pct_Bad'].sum():.1f}% of all bad accounts")
print(f"\nCharts saved:")
for f in ["monthly_bad_rate", "iv_chart",
          "correlation_matrix", "model_evaluation",
          "score_deciles", "shap_importance"]:
    print(f"  {f}.png")
