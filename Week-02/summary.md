# ⚡ Week 02 Summary
**Chapter:** 3 | **Date:** April 2026

---

## What I Learned This Week
- NumPy: creating arrays, slicing, vectorized operations, broadcasting, statistical functions
- Pandas: Series vs DataFrame, loading CSV, inspecting data, filtering, groupby, missing data
- Data Visualization: line, bar, histogram, scatter (Matplotlib) + boxplot, heatmap, pairplot (Seaborn)

## Toughest Concept This Week
**Broadcasting** in NumPy — understanding how a scalar or smaller array automatically extends to match a larger array's shape.

## Most Useful Thing Learned
`df.groupby().agg()` — being able to get mean, min, max, count all at once per group is incredibly powerful for real data analysis.

## Most Important Plot Learned
**Heatmap** of correlations — instantly shows which features are related to each other. Essential before building any ML model.

## Biggest Mistake to Avoid
Forgetting to handle missing data before training a model. Most ML algorithms will crash or give wrong results if NaN values are present.

## Next Week Plan
- Chapter 4: Probability & Statistics (deep dive)
- Practice: Load a real dataset, do full EDA with all visualization types
