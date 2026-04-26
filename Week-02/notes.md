# 📖 Week 02 Notes — NumPy, Pandas & Data Visualization
**Chapter:** 3 | **Course:** Samsung Innovation Campus AI

---

## Chapter 3: Exploratory Data Analysis

> 🧠 **Why this matters:** In real ML projects, 70–80% of your time is spent on data preparation — loading, cleaning, exploring, and visualising. NumPy and Pandas are the two most important tools for this.

---

## Part A: NumPy — Numerical Python

### 🔷 3.1 What is NumPy?
NumPy (Numerical Python) is the **core library** for numerical computation in Python. Almost all ML and data science libraries (Pandas, Scikit-learn, TensorFlow) are built on top of NumPy.

**Why use NumPy over regular Python lists?**
- **Much faster** — uses optimised C code under the hood
- **Memory efficient** — stores data in compact, typed arrays
- **Vectorized operations** — do math on entire arrays at once, no loops needed

---

### 🔷 3.2 Creating Arrays

```python
import numpy as np

# 1D array (vector)
arr = np.array([1, 2, 3, 4, 5])

# 2D array (matrix)
mat = np.array([[1, 2, 3],
                [4, 5, 6]])

# Helper functions
np.zeros(5)              # [0. 0. 0. 0. 0.]
np.ones((2, 3))          # 2×3 matrix of ones
np.arange(0, 10, 2)      # [0, 2, 4, 6, 8]
np.linspace(0, 1, 5)     # [0.0, 0.25, 0.5, 0.75, 1.0]
np.random.rand(3, 3)     # 3×3 random float matrix
np.eye(3)                # 3×3 identity matrix
```

---

### 🔷 3.3 Array Attributes

```python
arr = np.array([[1, 2, 3], [4, 5, 6]])

arr.shape    # (2, 3)  → 2 rows, 3 columns
arr.ndim     # 2       → number of dimensions
arr.dtype    # int64   → data type
arr.size     # 6       → total elements
```

> ⚠️ **Important:** Shape mismatches are the most common error in ML code. Always check `.shape` before operations!

---

### 🔷 3.4 Indexing and Slicing

```python
arr = np.array([[1, 2, 3],
                [4, 5, 6],
                [7, 8, 9]])

# Single element
arr[0, 1]      # → 2  (row 0, col 1)

# Entire row
arr[1, :]      # → [4, 5, 6]  (second row)

# Entire column
arr[:, 2]      # → [3, 6, 9]  (third column)

# Sub-matrix
arr[0:2, 1:3]  # → [[2,3],[5,6]]

# 1D slicing
a = np.array([10, 20, 30, 40, 50])
a[1:4]         # → [20, 30, 40]
a[-1]          # → 50 (last element)
```

---

### 🔷 3.5 Vectorized Operations & Broadcasting

**Vectorization** = operate on entire arrays at once (no Python loops)

```python
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

a + b      # → [5, 7, 9]
a * b      # → [4, 10, 18]
a ** 2     # → [1, 4, 9]
a + 10     # → [11, 12, 13]  ← broadcasting: scalar added to every element
```

> 🧠 **Analogy:** Instead of looping "for each student, multiply score by 2", vectorization does it all at once — like multiplying the whole column in Excel instantly.

**Broadcasting** = NumPy automatically extends smaller arrays to match larger ones:
```python
matrix = np.array([[1, 2, 3], [4, 5, 6]])
matrix + 100        # adds 100 to every element
matrix * np.array([1, 2, 3])  # multiplies each column by 1, 2, 3
```

---

### 🔷 3.6 Statistical Functions

```python
arr = np.array([10, 20, 30, 40, 50])

arr.sum()      # 150
arr.mean()     # 30.0
arr.min()      # 10
arr.max()      # 50
arr.std()      # standard deviation
arr.var()      # variance

# Axis operations (for 2D)
mat.sum(axis=0)   # sum of each column
mat.sum(axis=1)   # sum of each row
mat.mean(axis=0)  # mean of each column
```

---

### 🔷 3.7 Reshaping and Stacking

```python
arr = np.arange(1, 7)        # [1, 2, 3, 4, 5, 6]
arr.reshape(2, 3)            # → [[1,2,3],[4,5,6]]
arr.reshape(3, 2)            # → [[1,2],[3,4],[5,6]]

# Stacking
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
np.vstack([a, b])   # vertical stack → [[1,2,3],[4,5,6]]
np.hstack([a, b])   # horizontal stack → [1,2,3,4,5,6]
```

---

## Part B: Pandas — Tabular Data Analysis

### 🔷 3.8 What is Pandas?
Pandas is a library for working with **labelled, tabular data** — like Excel or CSV files. Built on top of NumPy but adds row/column labels, I/O tools, and powerful data manipulation.

---

### 🔷 3.9 Series and DataFrame

```python
import pandas as pd

# Series → 1D labelled array (one column)
s = pd.Series([85, 92, 78], index=['Alice', 'Bob', 'Charlie'])

# DataFrame → 2D labelled table (full dataset)
df = pd.DataFrame({
    'Name':  ['Alice', 'Bob', 'Charlie'],
    'Score': [85, 92, 78],
    'Age':   [21, 25, 23]
})
```

---

### 🔷 3.10 Loading and Inspecting Data

```python
# Load data
df = pd.read_csv('data.csv')

# ALWAYS run these first on any new dataset:
df.head()        # first 5 rows
df.tail()        # last 5 rows
df.info()        # column names, types, non-null counts
df.describe()    # count, mean, std, min, max, quartiles
df.shape         # (rows, columns)
df.columns       # list of column names
df.dtypes        # data type of each column
```

> ⭐ **Rule:** Always run `head()`, `info()`, `describe()` before doing ANYTHING else with a new dataset!

---

### 🔷 3.11 Selecting and Filtering

```python
# Select single column → returns Series
df['Score']

# Select multiple columns → returns DataFrame
df[['Name', 'Score']]

# Filter rows by condition
df[df['Score'] > 80]            # students who scored above 80
df[df['Age'] >= 21]             # students aged 21+
df[(df['Score'] > 80) & (df['Age'] > 20)]  # multiple conditions

# Select by row index
df.iloc[0]         # first row (integer position)
df.iloc[0:3]       # first 3 rows
df.loc[0, 'Name']  # row 0, 'Name' column
```

---

### 🔷 3.12 Sorting

```python
df.sort_values('Score', ascending=False)    # highest score first
df.sort_values(['Age', 'Score'])            # sort by multiple columns
df.sort_values('Score').reset_index(drop=True)  # reset index after sort
```

---

### 🔷 3.13 Handling Missing Data

```python
df.isna()              # True where data is missing
df.isna().sum()        # count of missing values per column
df.dropna()            # drop ALL rows with any missing value
df.dropna(subset=['Score'])   # drop only if 'Score' is missing
df.fillna(0)           # fill all missing with 0
df.fillna(df.mean())   # fill numeric columns with their mean
df['Age'].fillna(df['Age'].median())  # fill with median (better for skewed data)
```

> 🧠 **Decision rule:** If less than 5% of data is missing → drop rows. If more → fill with mean/median. Never just ignore missing data!

---

### 🔷 3.14 Adding/Modifying Columns

```python
# Add a new column
df['Grade'] = df['Score'].apply(lambda x: 'A' if x >= 90 else ('B' if x >= 80 else 'C'))

# Modify existing column
df['Score'] = df['Score'] * 1.1    # give everyone 10% bonus

# Rename columns
df.rename(columns={'Score': 'Marks', 'Age': 'Years'}, inplace=True)
```

---

### 🔷 3.15 GroupBy and Aggregation

```python
# Mean score per department
df.groupby('Department')['Score'].mean()

# Multiple aggregations at once
df.groupby('Department')['Score'].agg(['mean', 'min', 'max', 'count'])

# Count per group
df.groupby('Grade')['Name'].count()
```

> 📌 **Example use:** "What is the average salary per city?" → `df.groupby('City')['Salary'].mean()`

---

## Part C: Data Visualization

### 🔷 3.16 Why Visualize?
Before building any ML model, always visualise your data to:
- Spot patterns and trends
- Identify outliers
- Understand distributions
- See relationships between features

---

### 🔷 3.17 Matplotlib — Basic Plotting

```python
import matplotlib.pyplot as plt

# LINE PLOT — trends over time
plt.plot(x, y, marker='o', color='blue', linewidth=2)
plt.title('Title')
plt.xlabel('X Label')
plt.ylabel('Y Label')
plt.grid(True)
plt.show()

# BAR CHART — compare categories
plt.bar(categories, values, color='coral')
plt.title('Category Comparison')
plt.show()

# HISTOGRAM — distribution of values
plt.hist(data, bins=10, color='green', edgecolor='black')
plt.title('Distribution')
plt.show()

# SCATTER PLOT — relationship between 2 variables
plt.scatter(x, y, color='red', s=100)
plt.title('X vs Y')
plt.show()
```

---

### 🔷 3.18 Seaborn — Statistical Visualization

```python
import seaborn as sns

# BOX PLOT — spread, median, quartiles, outliers per group
sns.boxplot(x='Department', y='Score', data=df)

# HEATMAP — correlation between features
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f')

# PAIR PLOT — all pairwise relationships at once
sns.pairplot(df, hue='target_column')

# COUNT PLOT — frequency of each category
sns.countplot(x='Grade', data=df)

# VIOLIN PLOT — like boxplot but also shows distribution shape
sns.violinplot(x='Department', y='Score', data=df)
```

---

### 🔷 3.19 Which Plot to Use When?

| Plot | Library | Use it when... |
|------|---------|----------------|
| Line Plot | Matplotlib | Showing trends over time |
| Bar Chart | Matplotlib | Comparing values across categories |
| Histogram | Matplotlib | Showing distribution of one variable |
| Scatter Plot | Matplotlib | Showing relationship between 2 numeric variables |
| Box Plot | Seaborn | Comparing spread and outliers across groups |
| Heatmap | Seaborn | Showing correlation between all features |
| Pair Plot | Seaborn | Exploring all pairwise feature relationships |
| Count Plot | Seaborn | Showing frequency of categorical values |

> ⭐ **Pro tip:** Always plot a heatmap of correlations before building an ML model — it shows you which features are related and which are redundant.

---

## 💡 Key Takeaways — Week 2
1. NumPy arrays are faster and more efficient than Python lists — always use them for math
2. `.shape` is your best friend — check it constantly to avoid shape mismatch errors
3. Vectorized operations > loops — never loop over array elements when NumPy can do it
4. Always run `head()`, `info()`, `describe()` first on any new dataset
5. Handle missing data — never ignore it. Drop or fill based on how much is missing
6. Use histograms to check distribution, scatter plots for relationships, heatmap for correlations
7. Seaborn's `pairplot()` is the fastest way to explore a new dataset visually
