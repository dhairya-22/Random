# ═══════════════════════════════════════════════════════
#  INDOSTAR — COMPLETE PIPELINE WITH CHAID
#  Bucket 1 → 1+ Roll Rate Prediction
#  All cells kept short for manual typing
# ═══════════════════════════════════════════════════════


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  SECTION 1 — LIBRARIES
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# CELL 1 — Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, roc_curve
from statsmodels.stats.outliers_influence import variance_inflation_factor
import xgboost as xgb
import shap
from CHAID import Tree

pd.set_option("display.max_columns", 50)
pd.set_option("display.float_format", "{:.4f}".format)
print("All libraries loaded")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  SECTION 2 — CONFIG
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# CELL 2 — File and column config
DATA_FILE = r"C:\Users\dhairya.misra\Downloads\Model Building Raw data_1.parquet"
MONTH_COL = "MONTH"
TARGET    = "TARGET_1to1PLUS"
print("Config set")


# CELL 3 — Month splits
TRAIN_MONTHS = [
    "JUL24","AUG24","SEP24","OCT24",
    "NOV24","DEC24","JAN25","FEB25",
    "MAR25","APR25","MAY25","JUN25",
    "JUL25","AUG25","SEP25","OCT25"
]
TEST_MONTHS = ["NOV25","DEC25","JAN26"]
print("Train months:", len(TRAIN_MONTHS))
print("Test  months:", TEST_MONTHS)


# CELL 4 — Columns to drop
DROP_COLS = [
    "LOAN_NUMBER",
    "CUSTOMERID",
    "TOTAL_POS",
    "RECOVERY_STATUS",
    "WRITE_OFF_STATUS",
    "STATUS",
    "DPD_NEXT_MONTH",
    "STATUS_NEXT_MONTH",
    "WOFF_STATUS_NEXT_MONTH",
    "BOUNCE_NEXT_MONTH",
    "ODBKT_NEXT_MONTH",
    "X_BUCKET_NEXT_MONTH",
    "TARGET_PDM",
    "TARGET_XBkt",
    "TARGET_2to2PLUS",
    "TARGET_3to3PLUS",
    "TARGET_GT3PLUS",
    "TARGET_GT3PLUS_RF",
    "BV_LOSS",
    "TPOS_90PLUS",
    "TPOS_60PLUS",
    "TPOS_30PLUS",
    "DRIVING_LICENCE_NUMBER",
    "NOMINEE_NAME",
    "X_Bucket"
]
print("Drop list ready:", len(DROP_COLS), "columns")


# CELL 5 — Selected variables for model
SELECTED_VARS = [
    "DPD",
    "OVER_DUE_POS",
    "DPD_Transitions_3M",
    "TrendDPD_3M",
    "MaxDPD_12M",
    "MaxDPD_24M",
    "Latest_DPD",
    "DPD_Std_6M",
    "DPD_Std_12M",
    "CountInBkt_1Plus_6M",
    "CountInBkt_1Plus_12M",
    "WeightedAvgBkt_6M",
    "EverInBkt_2Plus_12M",
    "BounceRate_6M",
    "BounceRate_12M",
    "ChronicBouncer",
    "WeightedBounceScore",
    "coll_count_L6M",
    "coll_prop_L12M",
    "coll_rw_score",
    "EarlyWarningFlag",
    "DPD_Slope_6M",
    "MOB",
    "pct_paid",
    "IRR",
    "APPLICANT_CIBIL_SCORE",
    "AGE"
]
print("Selected vars:", len(SELECTED_VARS))


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  SECTION 3 — DATA LOADING
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# CELL 6 — Load data
import time
print("Loading data...")
start = time.time()
df = pd.read_parquet(DATA_FILE, engine="pyarrow")
elapsed = round(time.time() - start, 1)
print(f"Loaded in {elapsed}s")
print(f"Shape  : {df.shape[0]:,} rows x {df.shape[1]} cols")
print(f"Memory : {df.memory_usage(deep=True).sum()/1024**2:.1f} MB")


# CELL 7 — Validate data
months_found = set(df[MONTH_COL].unique())
months_need  = set(TRAIN_MONTHS + TEST_MONTHS)
missing      = months_need - months_found
if missing:
    print("Months missing:", sorted(missing))
else:
    print("All months confirmed in data")

bad = df[TARGET].sum()
br  = df[TARGET].mean() * 100
print(f"Bad  : {bad:,}")
print(f"Good : {len(df)-bad:,}")
print(f"Bad Rate: {br:.2f}%")


# CELL 8 — Drop leakage and ID columns
before = df.shape[1]
df = df.drop(
    columns=[c for c in DROP_COLS if c in df.columns],
    errors="ignore"
)
after = df.shape[1]
print(f"Columns before drop: {before}")
print(f"Columns after  drop: {after}")
print(f"Dropped: {before - after} columns")


# CELL 9 — Confirm selected vars exist
present = [v for v in SELECTED_VARS if v in df.columns]
missing_v = [v for v in SELECTED_VARS
             if v not in df.columns]
print(f"Variables found   : {len(present)}")
if missing_v:
    print(f"Variables missing : {missing_v}")
else:
    print("All selected variables confirmed")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  SECTION 4 — EDA
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# CELL 10 — Missing values
miss = df.isnull().sum()
miss = miss[miss > 0].sort_values(ascending=False)
miss_pct = (miss / len(df) * 100).round(2)
miss_df = pd.DataFrame({
    "Missing": miss,
    "Pct"    : miss_pct
})
print(f"Columns with missing values: {len(miss_df)}")
print(miss_df.head(20))


# CELL 11 — Fill remaining missing values
fill_cols = [
    "APP_CIBIL_SCORE_V1",
    "CO_APP_CIBIL_SCORE_V1",
    "CYCLE_LATEST",
    "AGE",
    "APPLICANT_CIBIL_SCORE"
]
for col in fill_cols:
    if col in df.columns:
        df[col].fillna(
            df[col].median(), inplace=True
        )
        print(f"Filled: {col}")

# Fill remaining selected vars with -999
for col in present:
    if df[col].isnull().sum() > 0:
        df[col].fillna(-999, inplace=True)
        print(f"Filled with -999: {col}")

print("Missing value treatment done")


# CELL 12 — Monthly bad rate table
df["MONTH"] = pd.Categorical(
    df["MONTH"],
    categories=TRAIN_MONTHS + TEST_MONTHS,
    ordered=True
)
all_months = TRAIN_MONTHS + TEST_MONTHS
monthly = (
    df[df["MONTH"].isin(all_months)]
    .groupby("MONTH", observed=True)[TARGET]
    .agg(Volume="count", Bad="sum")
)
monthly["Bad_Rate"] = (
    monthly["Bad"] / monthly["Volume"] * 100
).round(2)
print(monthly)


# CELL 13 — Monthly bad rate chart
fig, ax1 = plt.subplots(figsize=(15, 5))
ax2 = ax1.twinx()
colors = [
    "#5B8DB8" if m in TRAIN_MONTHS
    else "#E8A87C"
    for m in monthly.index
]
ax1.bar(monthly.index, monthly["Volume"],
        color=colors, alpha=0.85)
ax2.plot(monthly.index, monthly["Bad_Rate"],
         "o-", color="#E63946", lw=2.5, ms=6)
ax1.axvline(x=15.5, color="yellow",
            ls="--", lw=2)
ax2.yaxis.set_major_formatter(
    mtick.PercentFormatter())
ax1.set_ylabel("Volume")
ax2.set_ylabel("Bad Rate %")
plt.title(
    "Monthly Bad Rate — Bucket 1 Population",
    fontweight="bold"
)
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig("01_monthly_bad_rate.png", dpi=150)
plt.show()
print("Saved: 01_monthly_bad_rate.png")


# CELL 14 — Key variable stats
key_cols = [
    c for c in [
        "DPD","OVER_DUE_POS","MaxDPD_12M",
        "DPD_Std_6M","BounceRate_12M",
        "MOB","AGE","APPLICANT_CIBIL_SCORE"
    ] if c in df.columns
]
print(df[key_cols].describe().round(2))


# CELL 15 — Bad rate by customer category
if "CUSTOMER_CATEGORY" in df.columns:
    cat = (
        df.groupby(
            "CUSTOMER_CATEGORY"
        )[TARGET]
        .agg(Volume="count", Bad="sum")
    )
    cat["Bad_Rate"] = (
        cat["Bad"] / cat["Volume"] * 100
    ).round(2)
    print(cat.sort_values(
        "Bad_Rate", ascending=False))


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  SECTION 5 — CHAID BINNING
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# CELL 16 — CHAID tree function
def run_chaid(df, col, target=TARGET,
              max_depth=4,
              min_parent=500,
              min_child=250):
    tmp = df[[col, target]].dropna().copy()
    tmp[col]    = tmp[col].astype(float)
    tmp[target] = tmp[target].astype(str)
    try:
        tree = Tree.from_pandas_df(
            tmp,
            cols={
                col   : "continuous",
                target: "ordinal"
            },
            dep_variable=target,
            max_depth=max_depth,
            min_parent_node_size=min_parent,
            min_child_node_size=min_child
        )
        return tree
    except Exception as e:
        print(f"CHAID failed for {col}: {e}")
        return None

print("CHAID function ready")


# CELL 17 — Extract CHAID bins into table
def chaid_table(df, col, target=TARGET):
    tree = run_chaid(df, col)
    if tree is None:
        return None
    rows = []
    for node in tree:
        if node.is_terminal:
            members = node.members_data[target]
            total   = len(members)
            bad     = sum(
                1 for v in members if v == "1"
            )
            good    = total - bad
            br      = round(
                bad / total * 100, 2
            ) if total > 0 else 0
            rows.append({
                "Bin"     : str(node.choices),
                "Total"   : total,
                "Bad"     : bad,
                "Good"    : good,
                "Bad_Rate": br
            })
    return pd.DataFrame(rows).sort_values(
        "Bin"
    ).reset_index(drop=True)

print("CHAID table function ready")


# CELL 18 — Run CHAID on all selected variables
chaid_results = {}

for col in present:
    print(f"\n{'='*45}")
    print(f"CHAID — {col}")
    print(f"{'='*45}")
    try:
        tbl = chaid_table(df, col)
        if tbl is not None and len(tbl) > 0:
            print(tbl[[
                "Bin","Total",
                "Bad","Bad_Rate"
            ]].to_string(index=False))
            chaid_results[col] = tbl
        else:
            print("No splits found")
    except Exception as e:
        print(f"Skipped: {e}")

print(f"\nSuccessful: {len(chaid_results)} variables")


# CELL 19 — CHAID bad rate chart function
def plot_chaid(tbl, col):
    if tbl is None or len(tbl) == 0:
        print(f"No bins for {col}")
        return
    fig, ax1 = plt.subplots(figsize=(10, 4))
    ax2 = ax1.twinx()
    ax1.bar(
        tbl["Bin"], tbl["Total"],
        color="#B5D5C5", label="Volume"
    )
    ax2.plot(
        tbl["Bin"], tbl["Bad_Rate"],
        "o-", color="#E63946",
        lw=2.2, ms=6
    )
    ax2.yaxis.set_major_formatter(
        mtick.PercentFormatter())
    ax1.set_ylabel("Count")
    ax2.set_ylabel("Bad Rate %")
    plt.title(
        f"CHAID Bad Rate — {col}",
        fontweight="bold"
    )
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(f"chaid_{col}.png", dpi=150)
    plt.show()
    print(f"Saved: chaid_{col}.png")

print("Plot function ready")


# CELL 20 — Plot CHAID charts for key variables
key_plot_vars = [
    "DPD", "MaxDPD_12M",
    "BounceRate_12M", "AGE",
    "APPLICANT_CIBIL_SCORE",
    "DPD_Std_6M"
]
for col in key_plot_vars:
    if col in chaid_results:
        plot_chaid(chaid_results[col], col)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  SECTION 6 — WOE AND IV USING CHAID BINS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# CELL 21 — WOE and IV function using CHAID bins
def compute_woe_iv(df, col, target=TARGET):
    tmp        = df[[col, target]].dropna().copy()
    total_bad  = tmp[target].sum()
    total_good = len(tmp) - total_bad

    if total_bad == 0 or total_good == 0:
        return pd.DataFrame(), 0.0

    # Use CHAID bins
    tree = run_chaid(df, col)
    if tree is not None:
        splits = []
        for node in tree:
            if hasattr(node, "split_value"):
                if node.split_value is not None:
                    splits.append(
                        float(node.split_value)
                    )
        splits = sorted(set(splits))

        if len(splits) > 0:
            bounds = (
                [-np.inf] + splits + [np.inf]
            )
            tmp["bin"] = pd.cut(
                tmp[col].astype(float),
                bins=bounds,
                duplicates="drop"
            )
        else:
            tmp["bin"] = pd.qcut(
                tmp[col], q=10,
                duplicates="drop"
            )
    else:
        tmp["bin"] = pd.qcut(
            tmp[col], q=10,
            duplicates="drop"
        )

    g = (
        tmp.groupby("bin", observed=True)[target]
        .agg(bad="sum", total="count")
        .assign(
            good=lambda x: x["total"] - x["bad"]
        )
    )

    g["dist_bad"]  = (
        (g["bad"]  + 0.5) / (total_bad  + 0.5)
    )
    g["dist_good"] = (
        (g["good"] + 0.5) / (total_good + 0.5)
    )
    g["woe"] = np.log(
        g["dist_good"] / g["dist_bad"]
    )
    g["iv"]  = (
        (g["dist_good"] - g["dist_bad"])
        * g["woe"]
    )

    iv = round(g["iv"].sum(), 4)
    return g.reset_index(), iv

print("WOE/IV function ready")


# CELL 22 — Compute IV for all selected variables
def iv_label(v):
    if v > 0.5:  return "Suspicious"
    if v > 0.3:  return "Strong"
    if v > 0.1:  return "Medium"
    if v > 0.02: return "Weak"
    return               "Useless"

iv_rows = []
for col in present:
    try:
        _, iv = compute_woe_iv(df, col)
        iv_rows.append({
            "Variable": col,
            "IV"      : iv,
            "Strength": iv_label(iv)
        })
        print(f"{col}: IV={iv} ({iv_label(iv)})")
    except Exception as e:
        print(f"Skipped {col}: {e}")

iv_table = pd.DataFrame(iv_rows).sort_values(
    "IV", ascending=False
).reset_index(drop=True)

print("\nFull IV Table:")
print(iv_table.to_string(index=False))


# CELL 23 — IV bar chart
plt.figure(figsize=(11, 8))
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
            lw=1.5, label="0.1 Medium")
plt.axvline(0.3, color="red", ls="--",
            lw=1.5, label="0.3 Strong")
plt.xlabel("Information Value (IV)")
plt.title(
    "IV — CHAID Bins (Selected Variables)",
    fontweight="bold"
)
plt.legend()
plt.tight_layout()
plt.savefig("02_iv_chart.png", dpi=150)
plt.show()
print("Saved: 02_iv_chart.png")


# CELL 24 — Filter variables by IV
confirmed_vars = iv_table[
    (iv_table["IV"] >= 0.02) &
    (iv_table["IV"] <= 0.5)
]["Variable"].tolist()

dropped_iv = iv_table[
    (iv_table["IV"] < 0.02) |
    (iv_table["IV"] > 0.5)
]["Variable"].tolist()

print(f"Variables kept   : {len(confirmed_vars)}")
print(f"Variables dropped: {dropped_iv}")
print(f"\nConfirmed list:")
print(confirmed_vars)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  SECTION 7 — FEATURE SELECTION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# CELL 25 — Correlation heatmap
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
plt.title(
    "Correlation Matrix — Confirmed Variables",
    fontweight="bold"
)
plt.tight_layout()
plt.savefig("03_correlation.png", dpi=150)
plt.show()
print("Saved: 03_correlation.png")


# CELL 26 — Remove highly correlated variables
def remove_correlated(df, features,
                      threshold=0.85):
    corr  = df[features].corr().abs()
    upper = corr.where(
        np.triu(
            np.ones(corr.shape), k=1
        ).astype(bool)
    )
    to_drop = [
        col for col in upper.columns
        if any(upper[col] > threshold)
    ]
    kept = [
        c for c in features
        if c not in to_drop
    ]
    print(f"Dropped (corr>{threshold}): {to_drop}")
    print(f"Kept: {len(kept)} variables")
    return kept

final_vars = remove_correlated(
    df, confirmed_vars
)
print("\nFinal variables:")
print(final_vars)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  SECTION 8 — TRAIN TEST SPLIT
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# CELL 27 — Split by month
train_df = df[
    df["MONTH"].isin(TRAIN_MONTHS)
].copy()
test_df  = df[
    df["MONTH"].isin(TEST_MONTHS)
].copy()

X_train = train_df[final_vars].fillna(-999)
y_train = train_df[TARGET]
X_test  = test_df[final_vars].fillna(-999)
y_test  = test_df[TARGET]

print("TRAIN:", X_train.shape,
      "Bad Rate:",
      round(y_train.mean()*100, 2), "%")
print("TEST :", X_test.shape,
      "Bad Rate:",
      round(y_test.mean()*100, 2), "%")
print("Train months:",
      sorted(train_df["MONTH"].unique()))
print("Test  months:",
      sorted(test_df["MONTH"].unique()))


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  SECTION 9 — LOGISTIC REGRESSION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# CELL 28 — Train Logistic Regression
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


# CELL 29 — Evaluate Logistic Regression
lr_prob_train = lr.predict_proba(X_tr_s)[:, 1]
lr_prob_test  = lr.predict_proba(X_te_s)[:, 1]

lr_auc  = roc_auc_score(y_test, lr_prob_test)
lr_gini = 2 * lr_auc - 1
fpr, tpr, _ = roc_curve(y_test, lr_prob_test)
lr_ks   = round(float(max(tpr - fpr)), 4)

print("── Logistic Regression ──")
print(f"AUC  : {lr_auc:.4f}")
print(f"Gini : {lr_gini:.4f}")
print(f"KS   : {lr_ks:.4f}")


# CELL 30 — LR coefficient table
coef_df = pd.DataFrame({
    "Variable"   : final_vars,
    "Coefficient": lr.coef_[0]
}).sort_values(
    "Coefficient", key=abs, ascending=False
)
print("Top 10 variables by importance (LR):")
print(coef_df.head(10).to_string(index=False))


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  SECTION 10 — XGBOOST
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# CELL 31 — Train XGBoost
spw = (
    (y_train == 0).sum() /
    max((y_train == 1).sum(), 1)
)
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


# CELL 32 — Evaluate XGBoost
xgb_prob_train = xgb_model.predict_proba(
    X_train)[:, 1]
xgb_prob_test  = xgb_model.predict_proba(
    X_test)[:, 1]

xgb_auc  = roc_auc_score(y_test, xgb_prob_test)
xgb_gini = 2 * xgb_auc - 1
fpr, tpr, _ = roc_curve(y_test, xgb_prob_test)
xgb_ks  = round(float(max(tpr - fpr)), 4)

print("── XGBoost ──")
print(f"AUC  : {xgb_auc:.4f}")
print(f"Gini : {xgb_gini:.4f}")
print(f"KS   : {xgb_ks:.4f}")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  SECTION 11 — EVALUATION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# CELL 33 — Model comparison table
comparison = pd.DataFrame({
    "Model": [
        "Logistic Regression",
        "XGBoost"
    ],
    "AUC"  : [
        round(lr_auc, 4),
        round(xgb_auc, 4)
    ],
    "Gini" : [
        round(lr_gini, 4),
        round(xgb_gini, 4)
    ],
    "KS"   : [lr_ks, xgb_ks]
})
print(comparison.to_string(index=False))
print()
print("Benchmarks:")
print("  KS   > 0.30 Good | > 0.40 Great")
print("  AUC  > 0.70 Good | > 0.75 Great")
print("  Gini > 0.40 Good | > 0.50 Great")


# CELL 34 — ROC and KS chart
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for name, prob, auc, ks in [
    ("LR",  lr_prob_test,  lr_auc,  lr_ks),
    ("XGB", xgb_prob_test, xgb_auc, xgb_ks)
]:
    fpr, tpr, _ = roc_curve(y_test, prob)
    axes[0].plot(
        fpr, tpr, lw=2,
        label=f"{name} AUC={auc:.3f} KS={ks:.3f}"
    )

axes[0].plot([0,1],[0,1],"k--",lw=1)
axes[0].set_xlabel("False Positive Rate")
axes[0].set_ylabel("True Positive Rate")
axes[0].set_title("ROC Curve")
axes[0].legend(fontsize=9)

t = pd.DataFrame({
    "y": y_test.values,
    "p": xgb_prob_test
}).sort_values("p")
t["cb"] = t["y"].cumsum() / t["y"].sum()
t["cg"] = (
    (1 - t["y"]).cumsum() /
    (1 - t["y"]).sum()
)
axes[1].plot(
    range(len(t)), t["cb"],
    color="#E63946", label="Cum Bad"
)
axes[1].plot(
    range(len(t)), t["cg"],
    color="#457B9D", label="Cum Good"
)
axes[1].fill_between(
    range(len(t)), t["cb"], t["cg"],
    alpha=0.12, color="green"
)
axes[1].set_title(
    f"KS Plot — XGBoost (KS={xgb_ks:.3f})"
)
axes[1].legend()
plt.suptitle(
    "Model Evaluation — TARGET_1to1PLUS",
    fontsize=13, fontweight="bold"
)
plt.tight_layout()
plt.savefig("04_model_evaluation.png", dpi=150)
plt.show()
print("Saved: 04_model_evaluation.png")


# CELL 35 — PSI check
def compute_psi(expected, actual, n=10):
    bins = np.percentile(
        expected, np.linspace(0, 100, n+1)
    )
    bins[0]  -= 1e-6
    bins[-1] += 1e-6
    e = np.clip(
        np.histogram(expected, bins)[0]
        / len(expected), 1e-6, None
    )
    a = np.clip(
        np.histogram(actual, bins)[0]
        / len(actual), 1e-6, None
    )
    psi = float(np.sum(
        (a - e) * np.log(a / e)
    ))
    if psi < 0.1:
        status = "Stable"
    elif psi < 0.25:
        status = "Monitor"
    else:
        status = "Drift"
    print(f"PSI={round(psi,4)} → {status}")
    return psi

print("── PSI ──")
print("Logistic Regression:")
compute_psi(lr_prob_train, lr_prob_test)
print("XGBoost:")
compute_psi(xgb_prob_train, xgb_prob_test)


# CELL 36 — Per month evaluation
print("── Per Month ──")
rows = []
for month in TEST_MONTHS:
    mask = test_df["MONTH"].values == month
    sub  = test_df[mask]
    prob = xgb_prob_test[mask]
    if len(sub) < 10:
        continue
    fpr, tpr, _ = roc_curve(
        sub[TARGET], prob
    )
    rows.append({
        "Month"   : month,
        "N"       : len(sub),
        "Bad_Rate": round(
            sub[TARGET].mean()*100, 2),
        "AUC"     : round(
            roc_auc_score(
                sub[TARGET], prob), 4),
        "KS"      : round(
            float(max(tpr - fpr)), 4)
    })
print(pd.DataFrame(rows).to_string(
    index=False))


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  SECTION 12 — SHAP
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# CELL 37 — SHAP importance
explainer   = shap.TreeExplainer(xgb_model)
shap_values = explainer.shap_values(X_test)

plt.figure(figsize=(10, 7))
shap.summary_plot(
    shap_values, X_test,
    feature_names=final_vars,
    plot_type="bar",
    show=False
)
plt.title(
    "Feature Importance — SHAP",
    fontweight="bold"
)
plt.tight_layout()
plt.savefig("05_shap.png", dpi=150)
plt.show()
print("Saved: 05_shap.png")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  SECTION 13 — SCORE DECILE ANALYSIS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# CELL 38 — Build decile table
test_out = test_df[["MONTH", TARGET]].copy()
test_out["Roll_Prob"] = xgb_prob_test

test_out["Decile"] = 10 - pd.qcut(
    test_out["Roll_Prob"],
    q=10,
    labels=False,
    duplicates="drop"
)

overall_br = test_out[TARGET].mean() * 100

deciles = (
    test_out.groupby("Decile")
    .agg(
        N   = (TARGET, "count"),
        Bad = (TARGET, "sum")
    )
    .assign(
        Bad_Rate = lambda x: (
            x["Bad"]/x["N"]*100
        ).round(2),
        Lift = lambda x: (
            x["Bad"]/x["N"]*100/overall_br
        ).round(2),
        Pct_Bad = lambda x: (
            x["Bad"]/x["Bad"].sum()*100
        ).round(2)
    )
)
deciles["Cum_Bad"] = deciles[
    "Pct_Bad"
].cumsum().round(2)

print(f"Overall Bad Rate: {overall_br:.2f}%\n")
print(deciles.to_string())


# CELL 39 — Decile chart
fig, ax1 = plt.subplots(figsize=(11, 5))
ax2 = ax1.twinx()

ax1.bar(
    deciles.index.astype(str),
    deciles["N"],
    color="#B5D5C5",
    label="Volume"
)
ax2.plot(
    deciles.index.astype(str),
    deciles["Bad_Rate"],
    "o-", color="#E63946",
    lw=2.5, ms=6
)
ax2.axhline(
    overall_br, color="#999",
    ls="--", lw=1.5,
    label=f"Overall {overall_br:.1f}%"
)
ax2.yaxis.set_major_formatter(
    mtick.PercentFormatter())
ax1.set_xlabel(
    "Decile (1 = Highest Risk)")
ax1.set_ylabel("Volume")
ax2.set_ylabel("Bad Rate %")
plt.title(
    "Score Decile — TARGET_1to1PLUS",
    fontweight="bold"
)
fig.legend(loc="upper right")
plt.tight_layout()
plt.savefig("06_deciles.png", dpi=150)
plt.show()
print("Saved: 06_deciles.png")


# CELL 40 — Final summary
print("="*50)
print("  PIPELINE COMPLETE")
print("="*50)
print(f"Data        : {df.shape[0]:,} rows")
print(f"Features    : {len(final_vars)}")
print(f"CHAID trees : {len(chaid_results)}")
print(f"Train rows  : {len(train_df):,}")
print(f"Test rows   : {len(test_df):,}")
print(f"\nLogistic Regression:")
print(f"  AUC={lr_auc:.4f} Gini={lr_gini:.4f}"
      f" KS={lr_ks:.4f}")
print(f"\nXGBoost:")
print(f"  AUC={xgb_auc:.4f} Gini={xgb_gini:.4f}"
      f" KS={xgb_ks:.4f}")
print(f"\nTop 3 deciles capture:")
top3 = deciles.loc[1:3,"Pct_Bad"].sum()
print(f"  {top3:.1f}% of all bad accounts")
print(f"\nAll charts saved:")
for f in [
    "01_monthly_bad_rate",
    "02_iv_chart",
    "03_correlation",
    "04_model_evaluation",
    "05_shap",
    "06_deciles"
]:
    print(f"  {f}.png")
