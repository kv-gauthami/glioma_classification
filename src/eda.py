"""
eda.py
Here we are performing Exploratory analysis of the Glioma Grading dataset.
Saves plots to the 'figures/' folder.
"""

import os
import matplotlib.pyplot as plt
import seaborn as sns
from data_loader import load_data, get_feature_cols, TARGET_COL

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(BASE_DIR, "..", "figures")
os.makedirs(FIG_DIR, exist_ok=True)

df = load_data()
features = get_feature_cols(df)
genes = [c for c in features if c not in ("Gender", "Age_at_diagnosis")]

# 1. Duplicate rows (worth checking with mostly-binary features)
print("Duplicate rows:", df.duplicated().sum())

# 2. Age summary per grade
print("\nAge at diagnosis by grade (0=LGG, 1=GBM):\n",
      df.groupby(TARGET_COL)["Age_at_diagnosis"].describe().round(2))

# 3. Mutation rate (%) per gene: overall and by grade
mut_rate = (df.groupby(TARGET_COL)[genes].mean().T * 100).round(1)
mut_rate.columns = ["LGG_%", "GBM_%"]
mut_rate["Overall_%"] = (df[genes].mean() * 100).round(1)
print("\nMutation rate per gene (%):\n",
      mut_rate.sort_values("Overall_%", ascending=False))

# --- Plots ---
# a) Class distribution
plt.figure(figsize=(5, 4))
sns.countplot(data=df, x=TARGET_COL, color="crimson")
plt.xticks([0, 1], ["LGG", "GBM"])
plt.title("Class distribution")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "class_distribution.png"), dpi=120)
plt.close()

# b) Age distribution by grade
plt.figure(figsize=(7, 4))
sns.histplot(data=df, x="Age_at_diagnosis", hue=TARGET_COL, kde=True, bins=30, palette=["orange", "green"])
plt.title("Age at diagnosis by grade (0=LGG, 1=GBM)")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "age_by_grade.png"), dpi=120)
plt.close()

# c) Mutation rate per gene, LGG vs GBM
mut_rate[["LGG_%", "GBM_%"]].sort_values("LGG_%").plot(kind="barh", figsize=(7, 8), color=["orange", "green"])
plt.xlabel("% of patients with mutation")
plt.title("Gene mutation rate by grade")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "mutation_rate_by_grade.png"), dpi=120)
plt.close()

# d) Correlation heatmap (features + target)
plt.figure(figsize=(11, 9))
sns.heatmap(df[features + [TARGET_COL]].corr(), cmap="Reds", center=0)
plt.title("Feature correlation")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "correlation_heatmap.png"), dpi=120)
plt.close()

print(f"\nPlots saved to {FIG_DIR}")