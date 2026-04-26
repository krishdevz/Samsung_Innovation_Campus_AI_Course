# Week 01 - Pandas Practice
# Samsung Innovation Campus - AI Course

import pandas as pd
import numpy as np

print("=" * 50)
print("Pandas Basics Practice")
print("=" * 50)

# --- Create a Sample Dataset ---
data = {
    'Name':       ['Alice', 'Bob', 'Charlie', 'Diana', 'Eve', 'Frank'],
    'Age':        [22, 25, 23, 28, 21, None],
    'Score':      [85, 92, 78, 95, 88, 70],
    'Department': ['CS', 'IT', 'CS', 'AI', 'IT', 'AI'],
    'Passed':     [True, True, True, True, True, False]
}

df = pd.DataFrame(data)

print("\n--- Dataset ---")
print(df)

# --- Basic Inspection ---
print("\n--- Info ---")
print(df.info())

print("\n--- Describe (summary stats) ---")
print(df.describe())

print("\n--- Shape:", df.shape)
print("Columns:", list(df.columns))

# --- Selecting Data ---
print("\n--- Selecting Columns ---")
print("Scores:\n", df['Score'])
print("\nName & Department:\n", df[['Name', 'Department']])

# --- Filtering ---
print("\n--- Filtering ---")
print("Score > 85:\n", df[df['Score'] > 85])
print("\nCS Department:\n", df[df['Department'] == 'CS'])

# --- Sorting ---
print("\n--- Sorting by Score (descending) ---")
print(df.sort_values('Score', ascending=False))

# --- Missing Data ---
print("\n--- Missing Data ---")
print("Missing values:\n", df.isna().sum())
df['Age'] = df['Age'].fillna(df['Age'].mean())
print("After filling Age with mean:\n", df['Age'])

# --- GroupBy & Aggregation ---
print("\n--- GroupBy Department ---")
print(df.groupby('Department')['Score'].mean())
print("\nCount per Department:")
print(df.groupby('Department')['Name'].count())

# --- Adding a New Column ---
print("\n--- Adding Grade Column ---")
df['Grade'] = df['Score'].apply(lambda x: 'A' if x >= 90 else ('B' if x >= 80 else 'C'))
print(df[['Name', 'Score', 'Grade']])

# --- Value Counts ---
print("\n--- Grade Distribution ---")
print(df['Grade'].value_counts())
