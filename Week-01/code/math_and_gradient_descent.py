# Week 01 - Math for Data Science
# Samsung Innovation Campus - AI Course
# Topics: Vectors, Matrices, Dot Product, Statistics, Normalization, Gradient Descent

import numpy as np

print("=" * 60)
print("WEEK 01 — Math for Data Science")
print("=" * 60)

# ──────────────────────────────────────────
# 1. SCALARS, VECTORS, MATRICES, TENSORS
# ──────────────────────────────────────────
print("\n📦 1. Data Structures")
print("-" * 40)

scalar = 36.6
print(f"Scalar:  {scalar}  (just a number)")

vector = np.array([21, 170, 88])   # Age, Height, Score
print(f"Vector:  {vector}  (one student's features)")

matrix = np.array([
    [21, 170, 88],
    [25, 165, 92],
    [19, 180, 76]
])
print(f"Matrix (3 students × 3 features):\n{matrix}")
print(f"Shape: {matrix.shape}")

# 3D tensor (like a colour image — but small)
tensor = np.random.randint(0, 256, size=(4, 4, 3))
print(f"Tensor shape (4×4 image, 3 RGB channels): {tensor.shape}")

# ──────────────────────────────────────────
# 2. DOT PRODUCT
# ──────────────────────────────────────────
print("\n📐 2. Dot Product")
print("-" * 40)

a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
dot = np.dot(a, b)
print(f"a = {a}")
print(f"b = {b}")
print(f"a · b = {dot}  (manual: 1×4 + 2×5 + 3×6 = 4+10+18 = 32)")

# Real ML example: linear prediction
features = np.array([5.0, 3.0, 2.0])   # e.g. rooms, age, distance
weights  = np.array([10.0, -2.0, -1.5]) # learned weights
bias     = 50.0
prediction = np.dot(features, weights) + bias
print(f"\nML Prediction Example:")
print(f"Features: {features}, Weights: {weights}, Bias: {bias}")
print(f"Prediction = dot(features, weights) + bias = {prediction}")

# ──────────────────────────────────────────
# 3. MATRIX OPERATIONS
# ──────────────────────────────────────────
print("\n🔢 3. Matrix Operations")
print("-" * 40)

A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

print(f"A =\n{A}")
print(f"B =\n{B}")
print(f"A × B (matrix multiply) =\n{np.dot(A, B)}")
print(f"A^T (transpose) =\n{A.T}")
print(f"A + B (element-wise add) =\n{A + B}")

# ──────────────────────────────────────────
# 4. DESCRIPTIVE STATISTICS
# ──────────────────────────────────────────
print("\n📊 4. Descriptive Statistics")
print("-" * 40)

scores = np.array([55, 72, 68, 90, 85, 60, 95, 78, 82, 70])
print(f"Scores: {scores}")
print(f"Mean:        {np.mean(scores):.2f}")
print(f"Median:      {np.median(scores):.2f}")
print(f"Std Dev:     {np.std(scores):.2f}")
print(f"Variance:    {np.var(scores):.2f}")
print(f"Min:         {np.min(scores)}")
print(f"Max:         {np.max(scores)}")
print(f"Range:       {np.max(scores) - np.min(scores)}")

# Effect of outlier on mean vs median
scores_with_outlier = np.append(scores, 5)
print(f"\nWith outlier (5 added): {scores_with_outlier}")
print(f"Mean with outlier:   {np.mean(scores_with_outlier):.2f}  ← drops significantly")
print(f"Median with outlier: {np.median(scores_with_outlier):.2f}  ← barely changes")

# ──────────────────────────────────────────
# 5. NORMALIZATION vs STANDARDIZATION
# ──────────────────────────────────────────
print("\n⚖️ 5. Normalization vs Standardization")
print("-" * 40)

data = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
print(f"Original data: {data}")

# Min-Max Normalization → [0, 1]
normalized = (data - data.min()) / (data.max() - data.min())
print(f"Min-Max Normalized: {normalized}")

# Z-Score Standardization → mean=0, std=1
standardized = (data - data.mean()) / data.std()
print(f"Z-Score Standardized: {standardized.round(3)}")
print(f"  Mean after: {standardized.mean():.5f} (≈ 0)")
print(f"  Std after:  {standardized.std():.5f}  (≈ 1)")

# ──────────────────────────────────────────
# 6. GRADIENT DESCENT (from scratch!)
# ──────────────────────────────────────────
print("\n📉 6. Gradient Descent — Simple Demo")
print("-" * 40)
print("Finding the minimum of f(x) = x²  using gradient descent")
print("True minimum is at x = 0")

def f(x):
    return x ** 2

def gradient(x):
    return 2 * x   # derivative of x²

x = 10.0           # start far from minimum
learning_rate = 0.1
print(f"\nStarting at x = {x},  f(x) = {f(x)}")

for step in range(1, 11):
    grad = gradient(x)
    x = x - learning_rate * grad
    print(f"Step {step:2d}:  x = {x:8.4f},  f(x) = {f(x):.6f},  gradient = {grad:.4f}")

print(f"\nFinal x = {x:.6f}  (converged close to 0 ✅)")
