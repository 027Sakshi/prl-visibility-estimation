"""
============================================================
COMPARISON PLOTS
============================================================
"""

import os
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# PATHS
# ============================================================

INPUT = "results/comparison/comparison_metrics.csv"

OUTPUT = "results/comparison"

os.makedirs(OUTPUT, exist_ok=True)

print("=" * 70)
print("GENERATING COMPARISON PLOTS")
print("=" * 70)

# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(INPUT)

print("\nComparison Table\n")
print(df)

# ============================================================
# MAE
# ============================================================

plt.figure(figsize=(7,5))

plt.bar(df["Model"], df["MAE"])

plt.title("Model Comparison - MAE")

plt.ylabel("MAE")

plt.grid(axis="y")

plt.savefig(

    os.path.join(OUTPUT,"comparison_mae.png"),

    dpi=300,

    bbox_inches="tight"

)

plt.close()

# ============================================================
# RMSE
# ============================================================

plt.figure(figsize=(7,5))

plt.bar(df["Model"], df["RMSE"])

plt.title("Model Comparison - RMSE")

plt.ylabel("RMSE")

plt.grid(axis="y")

plt.savefig(

    os.path.join(OUTPUT,"comparison_rmse.png"),

    dpi=300,

    bbox_inches="tight"

)

plt.close()

# ============================================================
# R2
# ============================================================

plt.figure(figsize=(7,5))

plt.bar(df["Model"], df["R2"])

plt.title("Model Comparison - R²")

plt.ylabel("R²")

plt.grid(axis="y")

plt.savefig(

    os.path.join(OUTPUT,"comparison_r2.png"),

    dpi=300,

    bbox_inches="tight"

)

plt.close()

# ============================================================
# COMBINED
# ============================================================

plt.figure(figsize=(9,5))

plt.plot(df["Model"],df["MAE"],marker="o",label="MAE")

plt.plot(df["Model"],df["RMSE"],marker="o",label="RMSE")

plt.plot(df["Model"],df["R2"],marker="o",label="R²")

plt.title("Overall Model Comparison")

plt.legend()

plt.grid(True)

plt.savefig(

    os.path.join(OUTPUT,"comparison_combined.png"),

    dpi=300,

    bbox_inches="tight"

)

plt.close()

# ============================================================
# BEST MODEL
# ============================================================

best = df.sort_values(

    by="R2",

    ascending=False

).iloc[0]

with open(

    os.path.join(OUTPUT,"best_model.txt"),

    "w"

) as f:

    f.write("BEST MODEL\n")

    f.write("="*40+"\n\n")

    f.write(f"Model : {best['Model']}\n")

    f.write(f"MAE   : {best['MAE']:.4f}\n")

    f.write(f"RMSE  : {best['RMSE']:.4f}\n")

    f.write(f"R2    : {best['R2']:.4f}\n")

print("\nPlots Generated Successfully!")

print("\nFiles Created:")

print("comparison_mae.png")

print("comparison_rmse.png")

print("comparison_r2.png")

print("comparison_combined.png")

print("best_model.txt")

print("\nSTATUS : SUCCESS")