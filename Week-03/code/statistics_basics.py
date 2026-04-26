# Week 03 - Statistics Practice
# Samsung Innovation Campus - AI Course

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

print("=" * 60)
print("WEEK 03 — Statistics Fundamentals")
print("=" * 60)

# ──────────────────────────────────────────
# 1. POPULATION VS SAMPLE
# ──────────────────────────────────────────
print("\n🌍 1. Population vs Sample")
print("-" * 40)

np.random.seed(42)

# Simulate "true" population of 10,000 students
population = np.random.normal(loc=70, scale=12, size=10000)
print(f"Population (10,000 students):")
print(f"  True Mean:    {population.mean():.4f}")
print(f"  True Std Dev: {population.std():.4f}")

# Take multiple samples and see how close they are
print(f"\nSamples drawn from population:")
for i, n in enumerate([10, 50, 100, 500]):
    sample = np.random.choice(population, size=n, replace=False)
    error = abs(sample.mean() - population.mean())
    print(f"  Sample n={n:4d}: mean={sample.mean():.2f}, std={sample.std():.2f}, error={error:.2f}")

print(f"\n→ Larger samples give estimates closer to the true population values!")

# ──────────────────────────────────────────
# 2. DESCRIPTIVE STATISTICS DEEP DIVE
# ──────────────────────────────────────────
print("\n📊 2. Descriptive Statistics")
print("-" * 40)

scores = np.array([55, 72, 68, 90, 85, 60, 95, 78, 82, 70, 88, 45, 91, 76, 83])
print(f"Scores: {scores}")
print(f"\nMeasures of Central Tendency:")
print(f"  Mean:   {np.mean(scores):.2f}")
print(f"  Median: {np.median(scores):.2f}")
mode_result = stats.mode(scores, keepdims=True)
print(f"  Mode:   {mode_result.mode[0]}")

print(f"\nMeasures of Spread:")
print(f"  Range:         {np.max(scores) - np.min(scores)}")
print(f"  Variance:      {np.var(scores):.2f}")
print(f"  Std Deviation: {np.std(scores):.2f}")
print(f"  IQR:           {np.percentile(scores, 75) - np.percentile(scores, 25):.2f}")

print(f"\nPercentiles:")
print(f"  25th (Q1): {np.percentile(scores, 25)}")
print(f"  50th (Q2): {np.percentile(scores, 50)}")
print(f"  75th (Q3): {np.percentile(scores, 75)}")
print(f"  90th:      {np.percentile(scores, 90)}")

# Effect of outlier
scores_with_outlier = np.append(scores, 5)
print(f"\nWith outlier (score=5 added):")
print(f"  Mean before:   {np.mean(scores):.2f}   After: {np.mean(scores_with_outlier):.2f}  ← BIG change!")
print(f"  Median before: {np.median(scores):.2f}   After: {np.median(scores_with_outlier):.2f}  ← Small change")
print(f"→ Always use MEDIAN when data has outliers!")

# ──────────────────────────────────────────
# 3. HYPOTHESIS TESTING
# ──────────────────────────────────────────
print("\n🔬 3. Hypothesis Testing")
print("-" * 40)
print("Scenario: Did a new teaching method improve scores?")
print("H₀: New method has NO effect (mean score = 70)")
print("H₁: New method IMPROVES scores (mean score > 70)")
print("Significance level α = 0.05")

np.random.seed(10)
# Old method scores (known mean = 70)
old_scores = np.random.normal(70, 12, 30)

# New method scores (slightly higher)
new_scores = np.random.normal(76, 12, 30)

print(f"\nOld method: n={len(old_scores)}, mean={old_scores.mean():.2f}, std={old_scores.std():.2f}")
print(f"New method: n={len(new_scores)}, mean={new_scores.mean():.2f}, std={new_scores.std():.2f}")

# Two-sample t-test
t_stat, p_value = stats.ttest_ind(new_scores, old_scores, alternative='greater')
print(f"\nt-statistic: {t_stat:.4f}")
print(f"p-value:     {p_value:.4f}")

if p_value < 0.05:
    print(f"\n✅ p-value ({p_value:.4f}) < 0.05 → REJECT H₀")
    print(f"   The new teaching method SIGNIFICANTLY improves scores!")
else:
    print(f"\n❌ p-value ({p_value:.4f}) ≥ 0.05 → FAIL TO REJECT H₀")
    print(f"   No significant evidence that new method improves scores.")

# One-sample t-test: is our class mean different from national average?
print(f"\n--- One-Sample t-test ---")
print(f"Is our class mean significantly different from national average of 70?")
our_class = np.array([75, 82, 68, 91, 85, 79, 88, 72, 95, 77])
t_stat2, p_value2 = stats.ttest_1samp(our_class, popmean=70)
print(f"Our class mean: {our_class.mean():.2f}")
print(f"t-statistic: {t_stat2:.4f}, p-value: {p_value2:.4f}")
if p_value2 < 0.05:
    print(f"✅ Significant difference from national average!")
else:
    print(f"❌ No significant difference from national average.")

# ──────────────────────────────────────────
# 4. CONFIDENCE INTERVALS
# ──────────────────────────────────────────
print("\n📏 4. Confidence Intervals")
print("-" * 40)

sample = np.random.normal(70, 12, 50)
confidence = 0.95
ci = stats.t.interval(confidence,
                       df=len(sample)-1,
                       loc=np.mean(sample),
                       scale=stats.sem(sample))

print(f"Sample: n={len(sample)}, mean={sample.mean():.2f}, std={sample.std():.2f}")
print(f"\n95% Confidence Interval: ({ci[0]:.2f},  {ci[1]:.2f})")
print(f"→ We are 95% confident the true population mean lies between {ci[0]:.2f} and {ci[1]:.2f}")

# Show how CI width changes with sample size
print(f"\nHow sample size affects CI width:")
for n in [10, 30, 50, 100, 500]:
    s = np.random.normal(70, 12, n)
    ci_n = stats.t.interval(0.95, df=n-1, loc=np.mean(s), scale=stats.sem(s))
    width = ci_n[1] - ci_n[0]
    print(f"  n={n:4d}: CI width = {width:.2f}  ← {'narrow (precise)' if width < 5 else 'wide (imprecise)'}")
print(f"→ Larger samples give narrower, more precise confidence intervals!")

# ──────────────────────────────────────────
# 5. CORRELATION
# ──────────────────────────────────────────
print("\n📈 5. Correlation")
print("-" * 40)

np.random.seed(5)
hours_studied = np.random.uniform(1, 10, 50)
# Score increases with hours + some noise
score = 50 + 5 * hours_studied + np.random.normal(0, 5, 50)
age = np.random.uniform(18, 28, 50)         # random, no real correlation
sleep = 10 - 0.3 * hours_studied + np.random.normal(0, 1, 50)  # slight negative

df = pd.DataFrame({
    'Hours_Studied': hours_studied,
    'Score': score,
    'Age': age,
    'Sleep_Hours': sleep
})

print("Correlation with Score:")
corr_matrix = df.corr()
for col in ['Hours_Studied', 'Age', 'Sleep_Hours']:
    r = corr_matrix.loc['Score', col]
    strength = ('Strong' if abs(r) > 0.7 else ('Moderate' if abs(r) > 0.4 else 'Weak'))
    direction = 'positive' if r > 0 else 'negative'
    print(f"  Score vs {col:<15}: r = {r:+.3f}  → {strength} {direction} correlation")

# Heatmap
plt.figure(figsize=(7, 5))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.3f',
            linewidths=0.5, vmin=-1, vmax=1)
plt.title('Correlation Heatmap\nWhere do features relate to Score?',
          fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('plot_correlation_heatmap.png', dpi=100)
plt.show()
print("✅ Saved: plot_correlation_heatmap.png")

# Scatter plots to visualise correlations
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
pairs = [('Hours_Studied', 'steelblue'), ('Age', 'salmon'), ('Sleep_Hours', 'mediumseagreen')]
for ax, (col, color) in zip(axes, pairs):
    ax.scatter(df[col], df['Score'], color=color, alpha=0.7, edgecolors='black', linewidth=0.4)
    r = corr_matrix.loc['Score', col]
    ax.set_title(f'Score vs {col}\nr = {r:+.3f}', fontweight='bold')
    ax.set_xlabel(col)
    ax.set_ylabel('Score')
    ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('plot_correlation_scatter.png', dpi=100)
plt.show()
print("✅ Saved: plot_correlation_scatter.png")

print("\n" + "=" * 60)
print("IMPORTANT REMINDERS")
print("=" * 60)
print("1. p-value < 0.05 → statistically significant → reject H₀")
print("2. Larger sample → narrower confidence interval → more precise")
print("3. Correlation r close to ±1 → strong relationship")
print("4. Correlation r close to 0  → no linear relationship")
print("5. CORRELATION ≠ CAUSATION — always remember this!")
print("6. Use median (not mean) when data is skewed or has outliers")
