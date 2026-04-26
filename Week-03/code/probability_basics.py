# Week 03 - Probability Practice
# Samsung Innovation Campus - AI Course

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

print("=" * 60)
print("WEEK 03 — Probability Fundamentals")
print("=" * 60)

# ──────────────────────────────────────────
# 1. BASIC PROBABILITY
# ──────────────────────────────────────────
print("\n🎲 1. Basic Probability")
print("-" * 40)

# Simulating coin flips
np.random.seed(42)
flips = np.random.choice(['Heads', 'Tails'], size=10000)
heads_count = np.sum(flips == 'Heads')
print(f"10,000 coin flips:")
print(f"  Heads: {heads_count}  ({heads_count/10000:.3f})")
print(f"  Tails: {10000 - heads_count}  ({(10000-heads_count)/10000:.3f})")
print(f"  Theory says: 0.500  ← law of large numbers!")

# Simulating dice rolls
dice = np.random.randint(1, 7, size=60000)
print(f"\n60,000 dice rolls (each face should be ≈ 1/6 = 0.1667):")
for face in range(1, 7):
    count = np.sum(dice == face)
    print(f"  Face {face}: {count}  ({count/60000:.4f})")

# ──────────────────────────────────────────
# 2. PROBABILITY RULES
# ──────────────────────────────────────────
print("\n📏 2. Probability Rules")
print("-" * 40)

p_rain = 0.3
p_no_rain = 1 - p_rain
print(f"P(rain) = {p_rain}")
print(f"P(no rain) = 1 - {p_rain} = {p_no_rain}  [Complement Rule]")

p_a = 0.4   # P(passing Math)
p_b = 0.5   # P(passing English)
p_both = 0.2  # P(passing both)
p_either = p_a + p_b - p_both
print(f"\nP(pass Math) = {p_a}, P(pass English) = {p_b}, P(pass both) = {p_both}")
print(f"P(pass Math OR English) = {p_a} + {p_b} - {p_both} = {p_either}  [Addition Rule]")

# Independent events
p_head1 = 0.5
p_head2 = 0.5
p_two_heads = p_head1 * p_head2
print(f"\nP(Heads twice in a row) = {p_head1} × {p_head2} = {p_two_heads}  [Multiplication Rule]")

# ──────────────────────────────────────────
# 3. CONDITIONAL PROBABILITY
# ──────────────────────────────────────────
print("\n🔀 3. Conditional Probability")
print("-" * 40)

# Example: Students who pass given they studied
total_students = 100
studied = 60
passed_and_studied = 50
passed_without_studying = 10

p_studied = studied / total_students
p_passed_given_studied = passed_and_studied / studied
p_passed_given_not_studied = passed_without_studying / (total_students - studied)

print(f"Total students: {total_students}")
print(f"Students who studied: {studied}")
print(f"Passed AND studied: {passed_and_studied}")
print(f"\nP(studied) = {p_studied:.2f}")
print(f"P(pass | studied)     = {passed_and_studied}/{studied} = {p_passed_given_studied:.2f}  ← 83% pass rate!")
print(f"P(pass | not studied) = {passed_without_studying}/{total_students-studied} = {p_passed_given_not_studied:.2f}  ← 25% pass rate!")
print(f"\nConclusion: Studying makes you {p_passed_given_studied/p_passed_given_not_studied:.1f}x more likely to pass!")

# ──────────────────────────────────────────
# 4. BAYES' THEOREM
# ──────────────────────────────────────────
print("\n🔮 4. Bayes' Theorem")
print("-" * 40)
print("Medical test example:")
print("  Disease prevalence: 1% of population")
print("  Test sensitivity (true positive rate): 99%")
print("  Test specificity (true negative rate): 99%")
print("  Question: If you test POSITIVE, what's the real probability you have the disease?")

p_disease = 0.01           # prior: 1% have disease
p_no_disease = 0.99
p_positive_given_disease = 0.99    # test sensitivity
p_positive_given_no_disease = 0.01 # false positive rate

# Total probability of testing positive
p_positive = (p_positive_given_disease * p_disease +
              p_positive_given_no_disease * p_no_disease)

# Bayes' theorem
p_disease_given_positive = (p_positive_given_disease * p_disease) / p_positive

print(f"\n  P(disease | positive test) = {p_disease_given_positive:.4f} = {p_disease_given_positive*100:.1f}%")
print(f"  Surprising! Even with a 99% accurate test, only {p_disease_given_positive*100:.1f}% chance of actually having disease.")
print(f"  Why? Because disease is RARE (1%) — most positives are false alarms.")
print(f"  This is the BASE RATE FALLACY — always consider the prior probability!")

# ──────────────────────────────────────────
# 5. NORMAL DISTRIBUTION
# ──────────────────────────────────────────
print("\n🔔 5. Normal Distribution")
print("-" * 40)

# Generate normally distributed exam scores
np.random.seed(0)
mean_score = 70
std_score = 10
scores = np.random.normal(mean_score, std_score, 1000)

print(f"Generated 1000 exam scores: mean={mean_score}, std={std_score}")
print(f"Actual mean: {scores.mean():.2f}")
print(f"Actual std:  {scores.std():.2f}")

within_1std = np.sum(np.abs(scores - mean_score) <= std_score) / len(scores)
within_2std = np.sum(np.abs(scores - mean_score) <= 2*std_score) / len(scores)
within_3std = np.sum(np.abs(scores - mean_score) <= 3*std_score) / len(scores)

print(f"\n68-95-99.7 Rule verification:")
print(f"  Within 1 std dev (60-80): {within_1std:.3f}  ← theory: 0.683")
print(f"  Within 2 std dev (50-90): {within_2std:.3f}  ← theory: 0.954")
print(f"  Within 3 std dev (40-100): {within_3std:.3f}  ← theory: 0.997")

# Plot
plt.figure(figsize=(9, 5))
plt.hist(scores, bins=30, density=True, color='skyblue', edgecolor='black',
         linewidth=0.5, alpha=0.7, label='Simulated scores')
x = np.linspace(mean_score - 4*std_score, mean_score + 4*std_score, 200)
plt.plot(x, stats.norm.pdf(x, mean_score, std_score), 'r-', linewidth=2.5, label='Normal curve')
plt.axvline(mean_score, color='black', linewidth=2, linestyle='--', label=f'Mean = {mean_score}')
plt.axvline(mean_score + std_score, color='green', linewidth=1.5, linestyle=':', label=f'±1σ = {std_score}')
plt.axvline(mean_score - std_score, color='green', linewidth=1.5, linestyle=':')
plt.axvline(mean_score + 2*std_score, color='orange', linewidth=1.5, linestyle=':', label='±2σ')
plt.axvline(mean_score - 2*std_score, color='orange', linewidth=1.5, linestyle=':')
plt.title('Normal Distribution of Exam Scores\n(68-95-99.7 Rule)', fontsize=13, fontweight='bold')
plt.xlabel('Score')
plt.ylabel('Probability Density')
plt.legend()
plt.tight_layout()
plt.savefig('plot_normal_distribution.png', dpi=100)
plt.show()
print("✅ Saved: plot_normal_distribution.png")

# ──────────────────────────────────────────
# 6. SKEWED DISTRIBUTIONS
# ──────────────────────────────────────────
print("\n📈 6. Skewed Distributions")
print("-" * 40)

np.random.seed(1)
# Right-skewed: income data
income = np.random.exponential(scale=30000, size=1000)
print(f"Income data (right-skewed):")
print(f"  Mean:   ₹{income.mean():,.0f}  ← pulled right by high earners")
print(f"  Median: ₹{np.median(income):,.0f}  ← better representation of 'typical' person")
print(f"  Mean > Median = RIGHT skew confirmed")

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].hist(income, bins=40, color='salmon', edgecolor='black', linewidth=0.4)
axes[0].axvline(income.mean(), color='red', linewidth=2, label=f'Mean = ₹{income.mean():,.0f}')
axes[0].axvline(np.median(income), color='blue', linewidth=2, label=f'Median = ₹{np.median(income):,.0f}')
axes[0].set_title('Right-Skewed: Income Distribution', fontweight='bold')
axes[0].set_xlabel('Income (₹)')
axes[0].legend(fontsize=8)

# Normal for comparison
normal_data = np.random.normal(70, 10, 1000)
axes[1].hist(normal_data, bins=30, color='lightgreen', edgecolor='black', linewidth=0.4)
axes[1].axvline(normal_data.mean(), color='red', linewidth=2, label=f'Mean = {normal_data.mean():.1f}')
axes[1].axvline(np.median(normal_data), color='blue', linewidth=2, linestyle='--', label=f'Median = {np.median(normal_data):.1f}')
axes[1].set_title('Normal Distribution: Exam Scores', fontweight='bold')
axes[1].set_xlabel('Score')
axes[1].legend(fontsize=8)
plt.tight_layout()
plt.savefig('plot_skewed_vs_normal.png', dpi=100)
plt.show()
print("✅ Saved: plot_skewed_vs_normal.png")
