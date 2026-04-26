# Week 02 - Data Visualization Practice
# Samsung Innovation Campus - AI Course

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Dataset
data = {
    'Name':       ['Alice','Bob','Charlie','Diana','Eve','Frank','Grace','Henry'],
    'Age':        [22, 25, 23, 28, 21, 24, 24, 26],
    'Score':      [85, 92, 78, 95, 88, 70, 91, 65],
    'Department': ['CS','IT','CS','AI','IT','AI','CS','IT'],
    'Hours_Studied': [5, 7, 4, 9, 6, 3, 8, 3],
}
df = pd.DataFrame(data)

print("Dataset:")
print(df)
print("\nGenerating all plots...\n")

# ──────────────────────────────────────────
# 1. LINE PLOT — Score trend across students
# ──────────────────────────────────────────
plt.figure(figsize=(8, 4))
plt.plot(df['Name'], df['Score'], marker='o', color='steelblue',
         linewidth=2, markersize=8, markerfacecolor='white', markeredgewidth=2)
plt.title('Student Scores', fontsize=14, fontweight='bold')
plt.xlabel('Student')
plt.ylabel('Score')
plt.ylim(50, 100)
plt.grid(True, alpha=0.3)
plt.xticks(rotation=20)
plt.tight_layout()
plt.savefig('plot_01_line.png', dpi=100)
plt.show()
print("✅ Saved: plot_01_line.png — Use for: trends over a sequence")

# ──────────────────────────────────────────
# 2. BAR CHART — Score per student
# ──────────────────────────────────────────
colors = ['#2ECC71' if s >= 85 else '#E74C3C' for s in df['Score']]
plt.figure(figsize=(8, 4))
bars = plt.bar(df['Name'], df['Score'], color=colors, edgecolor='black', linewidth=0.7)
plt.axhline(y=85, color='navy', linestyle='--', linewidth=1.5, label='Pass threshold (85)')
plt.title('Score Comparison by Student', fontsize=14, fontweight='bold')
plt.xlabel('Student')
plt.ylabel('Score')
plt.legend()
plt.tight_layout()
plt.savefig('plot_02_bar.png', dpi=100)
plt.show()
print("✅ Saved: plot_02_bar.png — Use for: comparing categories")

# ──────────────────────────────────────────
# 3. HISTOGRAM — Score distribution
# ──────────────────────────────────────────
all_scores = [72, 85, 90, 68, 55, 92, 78, 88, 65, 95, 70, 82, 91, 76, 88, 60, 95, 67, 73, 89]
plt.figure(figsize=(8, 4))
plt.hist(all_scores, bins=8, color='mediumseagreen', edgecolor='black', linewidth=0.7)
plt.axvline(np.mean(all_scores), color='red', linestyle='--', linewidth=2, label=f'Mean = {np.mean(all_scores):.1f}')
plt.axvline(np.median(all_scores), color='blue', linestyle='--', linewidth=2, label=f'Median = {np.median(all_scores):.1f}')
plt.title('Score Distribution', fontsize=14, fontweight='bold')
plt.xlabel('Score')
plt.ylabel('Frequency')
plt.legend()
plt.tight_layout()
plt.savefig('plot_03_histogram.png', dpi=100)
plt.show()
print("✅ Saved: plot_03_histogram.png — Use for: distribution of one variable")

# ──────────────────────────────────────────
# 4. SCATTER PLOT — Hours studied vs Score
# ──────────────────────────────────────────
plt.figure(figsize=(7, 5))
scatter = plt.scatter(df['Hours_Studied'], df['Score'],
                      c=df['Score'], cmap='RdYlGn',
                      s=150, edgecolors='black', linewidth=0.7)
for _, row in df.iterrows():
    plt.annotate(row['Name'], (row['Hours_Studied'], row['Score']),
                 textcoords="offset points", xytext=(5, 5), fontsize=8)
plt.colorbar(scatter, label='Score')
plt.title('Hours Studied vs Score', fontsize=14, fontweight='bold')
plt.xlabel('Hours Studied per Day')
plt.ylabel('Score')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('plot_04_scatter.png', dpi=100)
plt.show()
print("✅ Saved: plot_04_scatter.png — Use for: relationship between 2 numeric vars")

# ──────────────────────────────────────────
# 5. BOX PLOT — Score by Department
# ──────────────────────────────────────────
plt.figure(figsize=(7, 5))
sns.boxplot(x='Department', y='Score', data=df, palette='pastel')
plt.title('Score Distribution by Department', fontsize=14, fontweight='bold')
plt.xlabel('Department')
plt.ylabel('Score')
plt.tight_layout()
plt.savefig('plot_05_boxplot.png', dpi=100)
plt.show()
print("✅ Saved: plot_05_boxplot.png — Use for: spread, median, outliers per group")

# ──────────────────────────────────────────
# 6. HEATMAP — Correlation matrix
# ──────────────────────────────────────────
numeric_df = df[['Age', 'Score', 'Hours_Studied']]
plt.figure(figsize=(6, 5))
sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm',
            fmt='.2f', linewidths=0.5, vmin=-1, vmax=1)
plt.title('Feature Correlation Heatmap', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('plot_06_heatmap.png', dpi=100)
plt.show()
print("✅ Saved: plot_06_heatmap.png — Use for: correlations between features")

# ──────────────────────────────────────────
# 7. COUNT PLOT — Students per Department
# ──────────────────────────────────────────
plt.figure(figsize=(6, 4))
sns.countplot(x='Department', data=df, palette='Set2')
plt.title('Students per Department', fontsize=14, fontweight='bold')
plt.xlabel('Department')
plt.ylabel('Count')
plt.tight_layout()
plt.savefig('plot_07_countplot.png', dpi=100)
plt.show()
print("✅ Saved: plot_07_countplot.png — Use for: frequency of categories")

print("\n🎉 All 7 plots saved successfully!")
print("\nQuick reference:")
print("  Line    → trends over time/sequence")
print("  Bar     → compare categories")
print("  Hist    → distribution of one variable")
print("  Scatter → relationship between 2 numeric variables")
print("  Box     → spread & outliers per group")
print("  Heatmap → correlation between features")
print("  Count   → frequency of categories")
