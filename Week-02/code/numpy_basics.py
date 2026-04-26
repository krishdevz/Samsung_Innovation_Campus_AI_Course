# Week 02 - NumPy Practice
# Samsung Innovation Campus - AI Course

import numpy as np

print("=" * 60)
print("WEEK 02 — NumPy Fundamentals")
print("=" * 60)

# ──────────────────────────────────────────
# 1. CREATING ARRAYS
# ──────────────────────────────────────────
print("\n📦 1. Creating Arrays")
print("-" * 40)

arr1 = np.array([1, 2, 3, 4, 5])
arr2 = np.array([[1, 2, 3], [4, 5, 6]])

print(f"1D Array:     {arr1}")
print(f"2D Array:\n{arr2}")
print(f"zeros(5):     {np.zeros(5)}")
print(f"ones(3,3):\n{np.ones((3,3))}")
print(f"arange:       {np.arange(0, 10, 2)}")
print(f"linspace:     {np.linspace(0, 1, 5)}")
print(f"identity:\n{np.eye(3)}")

# ──────────────────────────────────────────
# 2. ARRAY ATTRIBUTES
# ──────────────────────────────────────────
print("\n📋 2. Array Attributes")
print("-" * 40)

mat = np.array([[1, 2, 3], [4, 5, 6]])
print(f"Array:\n{mat}")
print(f"  .shape = {mat.shape}")
print(f"  .ndim  = {mat.ndim}")
print(f"  .dtype = {mat.dtype}")
print(f"  .size  = {mat.size}")

# ──────────────────────────────────────────
# 3. INDEXING AND SLICING
# ──────────────────────────────────────────
print("\n✂️ 3. Indexing and Slicing")
print("-" * 40)

arr = np.array([[1, 2, 3],
                [4, 5, 6],
                [7, 8, 9]])
print(f"Array:\n{arr}")
print(f"Element [1,2]:    {arr[1, 2]}")
print(f"Row 0:            {arr[0, :]}")
print(f"Column 1:         {arr[:, 1]}")
print(f"Sub-matrix [0:2, 1:3]:\n{arr[0:2, 1:3]}")

# ──────────────────────────────────────────
# 4. VECTORIZED OPERATIONS
# ──────────────────────────────────────────
print("\n⚡ 4. Vectorized Operations")
print("-" * 40)

a = np.array([1, 2, 3, 4, 5])
b = np.array([10, 20, 30, 40, 50])

print(f"a         = {a}")
print(f"b         = {b}")
print(f"a + b     = {a + b}")
print(f"a * b     = {a * b}")
print(f"a ** 2    = {a ** 2}")
print(f"a + 100   = {a + 100}   ← broadcasting scalar")
print(f"b / 10    = {b / 10}")
print(f"√a        = {np.sqrt(a).round(3)}")

# Compare speed: vectorized vs loop
import time
big = np.arange(1_000_000, dtype=float)

start = time.time()
result_np = big * 2
np_time = time.time() - start

start = time.time()
result_loop = [x * 2 for x in big]
loop_time = time.time() - start

print(f"\nSpeed comparison (1 million elements):")
print(f"  NumPy vectorized: {np_time*1000:.3f} ms")
print(f"  Python loop:      {loop_time*1000:.3f} ms")
print(f"  NumPy is {loop_time/np_time:.0f}x faster!")

# ──────────────────────────────────────────
# 5. STATISTICAL FUNCTIONS
# ──────────────────────────────────────────
print("\n📊 5. Statistical Functions")
print("-" * 40)

scores = np.array([72, 85, 90, 68, 55, 92, 78, 88, 65, 95])
print(f"Scores: {scores}")
print(f"Sum:         {scores.sum()}")
print(f"Mean:        {scores.mean():.2f}")
print(f"Median:      {np.median(scores):.2f}")
print(f"Std Dev:     {scores.std():.2f}")
print(f"Variance:    {scores.var():.2f}")
print(f"Min:         {scores.min()}")
print(f"Max:         {scores.max()}")

# Axis-based stats on 2D
data_2d = np.array([[10, 20, 30],
                    [40, 50, 60],
                    [70, 80, 90]])
print(f"\n2D array:\n{data_2d}")
print(f"Column means (axis=0): {data_2d.mean(axis=0)}")
print(f"Row means   (axis=1): {data_2d.mean(axis=1)}")

# ──────────────────────────────────────────
# 6. RESHAPE AND STACK
# ──────────────────────────────────────────
print("\n🔄 6. Reshape and Stack")
print("-" * 40)

flat = np.arange(1, 13)
print(f"Original (12 elements): {flat}")
print(f"Reshaped to (3,4):\n{flat.reshape(3, 4)}")
print(f"Reshaped to (2,6):\n{flat.reshape(2, 6)}")

a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
print(f"\nvstack:\n{np.vstack([a, b])}")
print(f"hstack: {np.hstack([a, b])}")

# ──────────────────────────────────────────
# 7. BOOLEAN INDEXING (very useful in ML!)
# ──────────────────────────────────────────
print("\n🎯 7. Boolean Indexing")
print("-" * 40)

scores = np.array([72, 85, 90, 68, 55, 92, 78, 88])
print(f"Scores: {scores}")
print(f"Scores > 80:          {scores[scores > 80]}")
print(f"Scores between 70-90: {scores[(scores >= 70) & (scores <= 90)]}")
print(f"Pass/fail (>= 75):    {scores >= 75}")
