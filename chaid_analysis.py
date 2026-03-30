# ═══════════════════════════════════════════════════════
#  CHAID ANALYSIS — SHORT CELLS (easy to type one by one)
#  For: IndoStar Bucket 1 → 1+ Roll Rate Model
# ═══════════════════════════════════════════════════════


# ── CHAID CELL 1 — Install library ─────────────────────
# Run this in TERMINAL first (not in notebook):
# pip install chaid
# If that doesn't work try:
# pip install CHAID

# Then run this cell to confirm install
import importlib
chaid_check = importlib.util.find_spec("CHAID")
if chaid_check:
    print("CHAID installed successfully")
else:
    print("CHAID not found — run pip install CHAID in terminal")


# ── CHAID CELL 2 — Import all needed libraries ──────────
from CHAID import Tree, Column
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

print("All imports done")


# ── CHAID CELL 3 — Pick variables for CHAID ─────────────
# These are the variables we will run CHAID on
# Start with the most important ones

chaid_vars = [
    "DPD",
    "MaxDPD_12M",
    "OVER_DUE_POS",
    "BounceRate_12M",
    "DPD_Std_6M",
    "CountInBkt_1Plus_12M",
    "WeightedAvgBkt_6M",
    "coll_rw_score",
    "MOB",
    "APPLICANT_CIBIL_SCORE",
    "AGE"
]

# Keep only variables that exist in your data
chaid_vars = [v for v in chaid_vars if v in df.columns]
print("CHAID variables confirmed:", len(chaid_vars))
print(chaid_vars)


# ── CHAID CELL 4 — Helper function for CHAID bins ───────
# This runs CHAID on ONE variable and returns optimal bins

def get_chaid_bins(df, col, target="TARGET_1to1PLUS",
                   max_depth=4, min_node=500):
    tmp = df[[col, target]].dropna().copy()
    tmp[col]    = tmp[col].astype(float)
    tmp[target] = tmp[target].astype(str)

    try:
        tree = Tree.from_pandas_df(
            tmp,
            cols={
                col    : "continuous",
                target : "ordinal"
            },
            dep_variable=target,
            max_depth=max_depth,
            min_parent_node_size=min_node,
            min_child_node_size=min_node // 2
        )
        return tree
    except Exception as e:
        print(f"CHAID failed for {col}: {e}")
        return None

print("CHAID bin function ready")


# ── CHAID CELL 5 — Run CHAID on DPD ─────────────────────
# Start with DPD — most important variable

print("="*50)
print("CHAID BINNING — DPD")
print("="*50)

tree_dpd = get_chaid_bins(df, "DPD")

if tree_dpd:
    tree_dpd.print_tree()


# ── CHAID CELL 6 — Run CHAID on MaxDPD_12M ──────────────
print("="*50)
print("CHAID BINNING — MaxDPD_12M")
print("="*50)

tree_maxdpd = get_chaid_bins(df, "MaxDPD_12M")

if tree_maxdpd:
    tree_maxdpd.print_tree()


# ── CHAID CELL 7 — Run CHAID on BounceRate_12M ──────────
print("="*50)
print("CHAID BINNING — BounceRate_12M")
print("="*50)

tree_bounce = get_chaid_bins(df, "BounceRate_12M")

if tree_bounce:
    tree_bounce.print_tree()


# ── CHAID CELL 8 — Run CHAID on AGE ─────────────────────
print("="*50)
print("CHAID BINNING — AGE")
print("="*50)

tree_age = get_chaid_bins(df, "AGE")

if tree_age:
    tree_age.print_tree()


# ── CHAID CELL 9 — Run all variables at once ────────────
# Loops through all chaid_vars and prints tree for each

chaid_trees = {}

for col in chaid_vars:
    print("\n" + "="*50)
    print(f"CHAID — {col}")
    print("="*50)
    try:
        tree = get_chaid_bins(df, col)
        if tree:
            tree.print_tree()
            chaid_trees[col] = tree
    except Exception as e:
        print(f"Skipped {col}: {e}")

print(f"\nSuccessful CHAID trees: {len(chaid_trees)}")


# ── CHAID CELL 10 — Extract bin splits and bad rates ────
# This converts the CHAID tree output into a clean table
# showing bin ranges and bad rates

def extract_chaid_table(df, col,
                        target="TARGET_1to1PLUS"):
    tmp  = df[[col, target]].dropna().copy()
    tree = get_chaid_bins(df, col)

    if tree is None:
        return None

    rows = []
    for node in tree:
        if node.is_terminal:
            # Get the split condition for this node
            split = str(node.choices)
            n     = node.members
            bad   = sum(
                1 for v in node.members_data[target]
                if v == "1"
            )
            good  = n - bad
            br    = round(bad / n * 100, 2) if n > 0 else 0
            rows.append({
                "Variable" : col,
                "Bin"      : split,
                "Total"    : n,
                "Bad"      : bad,
                "Good"     : good,
                "Bad_Rate" : br
            })

    out = pd.DataFrame(rows)
    return out

print("Extraction function ready")


# ── CHAID CELL 11 — Clean bad rate table for all vars ───
print("CHAID BAD RATE SUMMARY — ALL VARIABLES")
print("="*60)

all_chaid_tables = {}

for col in chaid_trees.keys():
    try:
        tbl = extract_chaid_table(df, col)
        if tbl is not None and len(tbl) > 0:
            print(f"\nVariable: {col}")
            print(tbl[["Bin","Total","Bad",
                        "Bad_Rate"]].to_string(index=False))
            all_chaid_tables[col] = tbl
    except Exception as e:
        print(f"Skipped {col}: {e}")


# ── CHAID CELL 12 — Plot CHAID bad rate chart ───────────
# Visual chart for each variable using CHAID bins

def plot_chaid_bins(tbl, col):
    if tbl is None or len(tbl) == 0:
        print(f"No data for {col}")
        return

    fig, ax1 = plt.subplots(figsize=(10, 4))
    ax2 = ax1.twinx()

    ax1.bar(tbl["Bin"], tbl["Total"],
            color="#B5D5C5", label="Volume")
    ax2.plot(tbl["Bin"], tbl["Bad_Rate"],
             "o-", color="#E63946", lw=2.2, ms=6)

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


# Plot for key variables
for col in ["DPD", "MaxDPD_12M",
            "BounceRate_12M", "AGE"]:
    if col in all_chaid_tables:
        plot_chaid_bins(all_chaid_tables[col], col)


# ── CHAID CELL 13 — CHAID vs Quantile comparison ────────
# Compare bad rates from CHAID bins vs quantile bins
# for the same variable — shows which is more meaningful

def compare_binning(df, col,
                    target="TARGET_1to1PLUS"):
    tmp = df[[col, target]].dropna().copy()
    n   = len(tmp)

    # Quantile bins
    tmp["qbin"] = pd.qcut(
        tmp[col], q=10, duplicates="drop"
    )
    q_br = (
        tmp.groupby("qbin", observed=True)[target]
        .agg(Total="count", Bad="sum")
        .assign(Bad_Rate=lambda x:
                (x["Bad"]/x["Total"]*100).round(2))
    )

    print(f"\n{'='*50}")
    print(f"Variable: {col}")
    print(f"{'='*50}")
    print("\nQuantile Binning (10 equal bins):")
    print(q_br[["Total","Bad",
                "Bad_Rate"]].to_string())

    print("\nCHAID Binning (statistically optimal):")
    tree = get_chaid_bins(df, col)
    if tree:
        tree.print_tree()


# Run comparison for top 3 variables
for col in ["DPD", "MaxDPD_12M", "APPLICANT_CIBIL_SCORE"]:
    if col in df.columns:
        compare_binning(df, col)


# ── CHAID CELL 14 — Use CHAID bins in WOE/IV ────────────
# Better IV using CHAID optimal splits instead of quantile

def compute_iv_chaid(df, col, n_bins=10,
                     target="TARGET_1to1PLUS"):
    tmp       = df[[col, target]].dropna().copy()
    total_bad = tmp[target].sum()
    total_good= len(tmp) - total_bad

    if total_bad == 0 or total_good == 0:
        return 0.0

    # Use CHAID tree to get bin boundaries
    tree = get_chaid_bins(df, col)

    if tree is None:
        # Fall back to quantile if CHAID fails
        try:
            tmp["bin"] = pd.qcut(
                tmp[col], q=n_bins,
                duplicates="drop"
            )
        except:
            tmp["bin"] = pd.cut(
                tmp[col], bins=n_bins,
                duplicates="drop"
            )
    else:
        # Use CHAID splits
        splits = []
        for node in tree:
            if hasattr(node, "split_value"):
                splits.append(node.split_value)
        splits = sorted(set(splits))

        if len(splits) > 0:
            boundaries = (
                [-np.inf] + splits + [np.inf]
            )
            tmp["bin"] = pd.cut(
                tmp[col],
                bins=boundaries,
                duplicates="drop"
            )
        else:
            tmp["bin"] = pd.qcut(
                tmp[col], q=n_bins,
                duplicates="drop"
            )

    g = (
        tmp.groupby("bin", observed=True)[target]
        .agg(bad="sum", total="count")
        .assign(good=lambda x: x["total"] - x["bad"])
    )

    g["db"]  = (g["bad"]  + 0.5) / (total_bad   + 0.5)
    g["dg"]  = (g["good"] + 0.5) / (total_good  + 0.5)
    g["woe"] = np.log(g["dg"] / g["db"])
    g["iv"]  = (g["dg"] - g["db"]) * g["woe"]

    return round(g["iv"].sum(), 4)

print("CHAID-based IV function ready")


# ── CHAID CELL 15 — CHAID IV table for all variables ────
print("IV USING CHAID BINS")
print("="*40)

chaid_iv_rows = []

for col in chaid_vars:
    try:
        iv = compute_iv_chaid(df, col)
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
        chaid_iv_rows.append({
            "Variable": col,
            "CHAID_IV": iv,
            "Strength": strength
        })
    except Exception as e:
        print(f"Skipped {col}: {e}")

chaid_iv_df = pd.DataFrame(chaid_iv_rows).sort_values(
    "CHAID_IV", ascending=False
).reset_index(drop=True)

print(chaid_iv_df.to_string(index=False))


# ── CHAID CELL 16 — Final summary ───────────────────────
print("="*55)
print("  CHAID ANALYSIS COMPLETE — SUMMARY")
print("="*55)
print(f"\nVariables analysed  : {len(chaid_vars)}")
print(f"Successful trees    : {len(chaid_trees)}")
print(f"\nKey findings:")
print(f"  CHAID finds optimal statistical split points")
print(f"  instead of equal frequency quantile bins")
print(f"  This gives more meaningful bad rate separation")
print(f"  and more reliable WOE/IV values")
print(f"\nCharts saved:")
for col in ["DPD","MaxDPD_12M",
            "BounceRate_12M","AGE"]:
    print(f"  chaid_{col}.png")
print(f"\nNext step: Use CHAID IV values to confirm")
print(f"your final variable selection for the model")
