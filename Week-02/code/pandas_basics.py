# Week 02 - Pandas Practice
# Samsung Innovation Campus - AI Course

import pandas as pd
import numpy as np

print("=" * 60)
print("WEEK 02 — Pandas Fundamentals")
print("=" * 60)

# ──────────────────────────────────────────
# 1. CREATE DATASET
# ──────────────────────────────────────────
print("\n📦 1. Creating a DataFrame")
print("-" * 40)

data = {
    'Name':       ['Alice', 'Bob', 'Charlie', 'Diana', 'Eve', 'Frank', 'Grace', 'Henry'],
    'Age':        [22, 25, 23, 28, 21, None, 24, 26],
    'Score':      [85, 92, 78, 95, 88, 70, 91, 65],
    'Department': ['CS', 'IT', 'CS', 'AI', 'IT', 'AI', 'CS', 'IT'],
    'Passed':     [True, True, True, True, True, False, True, False]
}

df = pd.DataFrame(data)
print(df)

# ──────────────────────────────────────────
# 2. INSPECTION
# ──────────────────────────────────────────
print("\n🔍 2. Data Inspection")
print("-" * 40)

print("Shape:", df.shape)
print("Columns:", list(df.columns))
print("\ndf.info():")
df.info()
print("\ndf.describe():")
print(df.describe())

# ──────────────────────────────────────────
# 3. SELECTING DATA
# ──────────────────────────────────────────
print("\n✂️ 3. Selecting Data")
print("-" * 40)

print("Single column (Score):")
print(df['Score'].values)

print("\nMultiple columns:")
print(df[['Name', 'Score', 'Department']])

print("\nRow by index (iloc[0]):")
print(df.iloc[0])

print("\nRows 2-4 (iloc[2:5]):")
print(df.iloc[2:5])

# ──────────────────────────────────────────
# 4. FILTERING
# ──────────────────────────────────────────
print("\n🎯 4. Filtering Rows")
print("-" * 40)

print("Score > 85:")
print(df[df['Score'] > 85][['Name', 'Score']])

print("\nCS Department:")
print(df[df['Department'] == 'CS'][['Name', 'Score']])

print("\nAI dept AND Score > 80:")
print(df[(df['Department'] == 'AI') & (df['Score'] > 80)][['Name', 'Score']])

print("\nPassed = False:")
print(df[df['Passed'] == False][['Name', 'Score']])

# ──────────────────────────────────────────
# 5. SORTING
# ──────────────────────────────────────────
print("\n🔢 5. Sorting")
print("-" * 40)

print("Top 3 by Score:")
print(df.sort_values('Score', ascending=False).head(3)[['Name', 'Score']])

print("\nSorted by Dept then Score:")
print(df.sort_values(['Department', 'Score'])[['Name', 'Department', 'Score']])

# ──────────────────────────────────────────
# 6. MISSING DATA
# ──────────────────────────────────────────
print("\n🔧 6. Handling Missing Data")
print("-" * 40)

print("Missing values per column:")
print(df.isna().sum())

print(f"\nAge before: {df['Age'].values}")
df['Age'] = df['Age'].fillna(df['Age'].mean())
print(f"Age after filling with mean: {df['Age'].values.round(1)}")

# ──────────────────────────────────────────
# 7. ADDING COLUMNS
# ──────────────────────────────────────────
print("\n➕ 7. Adding Columns")
print("-" * 40)

df['Grade'] = df['Score'].apply(lambda x: 'A' if x >= 90 else ('B' if x >= 80 else 'C'))
df['Score_Scaled'] = ((df['Score'] - df['Score'].min()) /
                      (df['Score'].max() - df['Score'].min())).round(3)

print(df[['Name', 'Score', 'Grade', 'Score_Scaled']])

# ──────────────────────────────────────────
# 8. GROUPBY & AGGREGATION
# ──────────────────────────────────────────
print("\n📊 8. GroupBy & Aggregation")
print("-" * 40)

print("Mean score per Department:")
print(df.groupby('Department')['Score'].mean().round(2))

print("\nMultiple stats per Department:")
print(df.groupby('Department')['Score'].agg(['mean', 'min', 'max', 'count']))

print("\nGrade distribution:")
print(df['Grade'].value_counts())

print("\nPass rate per Department:")
print(df.groupby('Department')['Passed'].mean().round(2))
