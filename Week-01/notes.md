# 📖 Week 01 Notes — Chapters 1, 2 & 3

**Course:** Samsung Innovation Campus — AI  
**Week:** 01 | **Date:** April 2026

---

## Chapter 1: Introduction to AI & Machine Learning

### Key Definitions
- **AI**: Systems that perform tasks requiring human-like intelligence (reasoning, learning, perception)
- **ML**: Subset of AI — systems that *learn from data* without explicit step-by-step programming
- **Deep Learning**: Subset of ML — uses multi-layered neural networks for complex patterns
- **Data Science**: Extracting insights from data using analysis, visualization, and modeling

### Hierarchy
```
AI  ⊇  Machine Learning  ⊇  Deep Learning
```

### Types of Machine Learning

| Type | Data | Goal | Example |
|------|------|------|---------|
| Supervised | Input + Labels | Predict output | Spam detection |
| Unsupervised | Input only | Find patterns | Customer segmentation |
| Reinforcement | Reward signals | Maximize reward | Game-playing agents |

### ML Pipeline
```
Data → Preprocessing → Feature Selection → Model → Train → Evaluate → Predict
```

---

## Chapter 2: Math for Data Science

### Data Structures
- **Scalar** → single number (e.g. 4)
- **Vector** → 1D array (e.g. [20, 170, 85])
- **Matrix** → 2D grid (dataset: rows=samples, cols=features)
- **Tensor** → 3D+ array (e.g. image: height × width × RGB)

### Key Operations
- **Dot Product**: `a · b = Σ(ai × bi)` — measures alignment of two vectors
- **Matrix Multiply**: `C[i][j] = Σ A[i][k] × B[k][j]` — used in every neural network
- **Transpose**: swap rows and columns → written as `A^T`

### Descriptive Statistics
```
Mean:     μ = Σx / n
Variance: σ² = Σ(x - μ)² / n
Std Dev:  σ = √σ²
```

### Normalization vs Standardization
```
Min-Max Normalization: X_scaled = (X - min) / (max - min)  → range [0,1]
Z-Score Standardization: X_scaled = (X - mean) / std       → centered at 0
```

### Gradient Descent (How ML Models Learn)
```
New Weight = Old Weight - (Learning Rate × Gradient)
```
- **Gradient** = derivative of loss function w.r.t. weights
- **Learning Rate** = step size (too large = overshoot, too small = slow)
- **Chain Rule**: `dz/dx = f'(g(x)) × g'(x)` — used in backpropagation

---

## Chapter 3: NumPy & Pandas

### NumPy Essentials
```python
import numpy as np
arr = np.array([1, 2, 3])          # 1D array
mat = np.array([[1,2],[3,4]])       # 2D array (matrix)

arr.shape   # dimensions
arr.dtype   # data type
arr.mean()  # average
arr.std()   # standard deviation

# Slicing
arr[1:4]      # elements 1,2,3
mat[0, :]     # entire first row
mat[:, 1]     # entire second column
```

### Pandas Essentials
```python
import pandas as pd
df = pd.read_csv('data.csv')    # load data

df.head()       # first 5 rows
df.info()       # column types + null counts
df.describe()   # summary statistics

df['age']                        # select column
df[df['score'] > 80]             # filter rows
df.fillna(df.mean())             # fill missing values
df.groupby('dept')['sal'].mean() # group + aggregate
```

### Data Visualization
```python
import matplotlib.pyplot as plt
import seaborn as sns

# Line plot — trends
plt.plot(x, y)

# Bar chart — categories
plt.bar(categories, values)

# Histogram — distribution
plt.hist(data, bins=10)

# Scatter plot — relationship
plt.scatter(x, y)

# Box plot — outliers per group
sns.boxplot(x='category', y='value', data=df)

# Heatmap — feature correlations
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')

# Pair plot — all relationships
sns.pairplot(df, hue='target')
```

---

## 💡 Key Takeaways This Week
1. AI learns from data instead of following hardcoded rules
2. Math (vectors, matrices, derivatives) is the language of ML
3. NumPy + Pandas are the two most important Python libraries for data
4. Always visualize data before building any model
5. Gradient Descent is how models improve — understand it deeply
