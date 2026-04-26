# Week 01 - Data Visualization Practice
# Samsung Innovation Campus - AI Course

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Sample dataset
data = {
    'Name':       ['Alice', 'Bob', 'Charlie', 'Diana', 'Eve', 'Frank'],
    'Age':        [22, 25, 23, 28, 21, 24],
    'Score':      [85, 92, 78, 95, 88, 70],
    'Department': ['CS', 'IT', 'CS', 'AI', 'IT', 'AI'],
}
df = pd.DataFrame(data)

# ─── 1. Line Plot ───
plt.figure(figsize=(7, 4))
plt.plot(df['Name'], df['Score'], marker='o', color='steelblue', linewidth=2)
plt.title('Student Scores')
plt.xlabel('Student')
plt.ylabel('Score')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('plot_1_line.png')
plt.show()
print("Saved: plot_1_line.png")

# ─── 2. Bar Chart ───
plt.figure(figsize=(7, 4))
plt.bar(df['Name'], df['Score'], color='coral', edgecolor='black')
plt.title('Score Comparison')
plt.xlabel('Student')
plt.ylabel('Score')
plt.tight_layout()
plt.savefig('plot_2_bar.png')
plt.show()
print("Saved: plot_2_bar.png")

# ─── 3. Histogram ───
scores = [85, 92, 78, 95, 88, 70, 82, 91, 76, 88, 95, 67, 73, 89]
plt.figure(figsize=(7, 4))
plt.hist(scores, bins=5, color='mediumseagreen', edgecolor='black')
plt.title('Score Distribution')
plt.xlabel('Score')
plt.ylabel('Frequency')
plt.tight_layout()
plt.savefig('plot_3_histogram.png')
plt.show()
print("Saved: plot_3_histogram.png")

# ─── 4. Scatter Plot ───
age =   [22, 25, 23, 28, 21, 24, 26, 27]
score = [85, 92, 78, 95, 88, 70, 83, 91]
plt.figure(figsize=(7, 4))
plt.scatter(age, score, color='mediumpurple', s=100, edgecolors='black')
plt.title('Age vs Score')
plt.xlabel('Age')
plt.ylabel('Score')
plt.tight_layout()
plt.savefig('plot_4_scatter.png')
plt.show()
print("Saved: plot_4_scatter.png")

# ─── 5. Box Plot ───
plt.figure(figsize=(7, 4))
sns.boxplot(x='Department', y='Score', data=df, palette='pastel')
plt.title('Score Distribution by Department')
plt.tight_layout()
plt.savefig('plot_5_boxplot.png')
plt.show()
print("Saved: plot_5_boxplot.png")

# ─── 6. Heatmap (Correlation) ───
num_df = df[['Age', 'Score']]
plt.figure(figsize=(5, 4))
sns.heatmap(num_df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Feature Correlation Heatmap')
plt.tight_layout()
plt.savefig('plot_6_heatmap.png')
plt.show()
print("Saved: plot_6_heatmap.png")

print("\nAll plots saved successfully!")
