# Week 01 - NumPy Practice
# Samsung Innovation Campus - AI Course

import numpy as np

print("=" * 50)
print("NumPy Basics Practice")
print("=" * 50)

# --- Creating Arrays ---
arr_1d = np.array([1, 2, 3, 4, 5])
arr_2d = np.array([[1, 2, 3], [4, 5, 6]])

print("\n1D Array:", arr_1d)
print("2D Array:\n", arr_2d)
print("Shape:", arr_2d.shape)
print("Dimensions:", arr_2d.ndim)
print("Data type:", arr_2d.dtype)

# --- Array Creation Functions ---
print("\nZeros:", np.zeros(5))
print("Ones:", np.ones((2, 3)))
print("Range:", np.arange(0, 10, 2))
print("Linspace:", np.linspace(0, 1, 5))

# --- Indexing and Slicing ---
print("\n--- Slicing ---")
print("First row:", arr_2d[0, :])
print("Second column:", arr_2d[:, 1])
print("Element [1][2]:", arr_2d[1, 2])

# --- Vectorized Operations ---
print("\n--- Vectorized Operations ---")
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

print("a + b =", a + b)
print("a * b =", a * b)
print("a ** 2 =", a ** 2)
print("a + 10 =", a + 10)          # broadcasting

# --- Dot Product ---
print("\n--- Dot Product ---")
dot = np.dot(a, b)
print(f"Dot product of {a} and {b} = {dot}")
# Manual check: 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32

# --- Statistical Functions ---
print("\n--- Statistics ---")
data = np.array([10, 20, 30, 40, 50])
print("Data:", data)
print("Sum:", data.sum())
print("Mean:", data.mean())
print("Std Dev:", data.std())
print("Min:", data.min())
print("Max:", data.max())

# --- Matrix Operations ---
print("\n--- Matrix Multiplication ---")
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])
C = np.dot(A, B)
print("A =\n", A)
print("B =\n", B)
print("A × B =\n", C)

# --- Reshape ---
print("\n--- Reshape ---")
flat = np.arange(1, 7)
reshaped = flat.reshape(2, 3)
print("Original:", flat)
print("Reshaped to (2,3):\n", reshaped)

# --- Normalization (Min-Max) ---
print("\n--- Min-Max Normalization ---")
raw = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
normalized = (raw - raw.min()) / (raw.max() - raw.min())
print("Original:", raw)
print("Normalized:", normalized)

# --- Standardization (Z-Score) ---
print("\n--- Z-Score Standardization ---")
standardized = (raw - raw.mean()) / raw.std()
print("Original:", raw)
print("Standardized:", standardized.round(2))
