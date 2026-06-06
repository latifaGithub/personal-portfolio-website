---
layout: post
title: "The Complete Data Cleaning & Preparation Checklist"
type: blog
tags: [Data Cleaning, Python, pandas, Data Science]
description: "A practical, step-by-step checklist for preparing your dataset before any modelling work -covering missing values, outliers, encoding, scaling, and data leakage."
cover: "linear-gradient(135deg, #0f2027 0%, #1a4a3a 50%, #2c5364 100%)"
cover_icon: "🧹"
---

Garbage in, garbage out. No model -however sophisticated -can compensate for a poorly prepared dataset. Data cleaning is often the difference between a model that performs well in production and one that fails silently.

This checklist covers the essential steps, in order, with practical Python snippets for each.

---

## Step 1 -Understand Your Data First

Before touching anything, spend time *looking* at the data. Cleaning without understanding leads to mistakes you won't catch until much later.

```python
import pandas as pd

df = pd.read_csv('data.csv')

print(df.shape)              # rows × columns
print(df.dtypes)             # data type per column
print(df.describe())         # statistical summary (numerical)
print(df.describe(include='object'))  # summary for categorical columns
print(df.head(10))           # first 10 rows
```

**Ask yourself:**
- What does each column represent? What are the expected ranges and categories?
- What is the target variable and how is it distributed?
- Are there any obvious issues visible immediately (e.g., dates stored as strings)?

---

## Step 2 -Handle Missing Values

Missing values are almost always present and must be addressed explicitly. Never let a model silently drop or zero-fill them without your decision.

```python
# Identify missing values
print(df.isnull().sum())
print(df.isnull().mean().sort_values(ascending=False) * 100)   # % missing per column
```

**Strategy by severity:**

| Missing % | Recommended Approach |
|---|---|
| < 5% | Impute (median / mode) or drop rows |
| 5–30% | Impute; add a `_missing` indicator column to preserve the signal |
| > 30% | Consider dropping the column; check if missingness itself is informative |

```python
from sklearn.impute import SimpleImputer

# Numerical: median imputation (robust to outliers)
imputer = SimpleImputer(strategy='median')
df['age'] = imputer.fit_transform(df[['age']])

# Categorical: most frequent value
df['city'].fillna(df['city'].mode()[0], inplace=True)

# Preserve missingness signal before imputing
df['income_was_missing'] = df['income'].isnull().astype(int)
```

---

## Step 3 -Remove Duplicates

Duplicate rows inflate your dataset and silently bias model training -especially in classification tasks where they can shift class distributions.

```python
print(f"Duplicate rows: {df.duplicated().sum()}")
df.drop_duplicates(inplace=True)
df.reset_index(drop=True, inplace=True)
```

Also check for **near-duplicates** -rows identical except for minor formatting differences (e.g., `"New York"` vs `"new york"` vs `"New York "`).

---

## Step 4 -Fix Data Types

Columns are frequently imported with the wrong type: numbers stored as strings, dates as objects, categories as unconstrained text.

```python
# Convert to correct types
df['signup_date'] = pd.to_datetime(df['signup_date'], errors='coerce')
df['price']       = pd.to_numeric(df['price'], errors='coerce')
df['segment']     = df['segment'].astype('category')

# Strip whitespace from all string columns
str_cols = df.select_dtypes(include='object').columns
df[str_cols] = df[str_cols].apply(lambda col: col.str.strip())

# Extract useful components from datetime columns
df['signup_year']  = df['signup_date'].dt.year
df['signup_month'] = df['signup_date'].dt.month
```

---

## Step 5 -Detect and Handle Outliers

Outliers can distort model parameters, especially in linear models, and lead to poor generalisation.

```python
import numpy as np

# IQR method -robust for non-normal distributions
Q1  = df['revenue'].quantile(0.25)
Q3  = df['revenue'].quantile(0.75)
IQR = Q3 - Q1

lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR

n_outliers = ((df['revenue'] < lower) | (df['revenue'] > upper)).sum()
print(f"Outliers detected: {n_outliers}")
```

**Handling options:**

| Method | When to Use |
|---|---|
| **Winsorise (cap)** | The outlier is valid data but extreme -retain the row, limit the value |
| **Log transform** | Right-skewed distribution (revenue, population) |
| **Remove** | Confirmed data errors, not real observations |

```python
# Winsorise
df['revenue'] = df['revenue'].clip(lower=lower, upper=upper)

# Log transform for skewed data (add 1 to handle zeros)
df['revenue_log'] = np.log1p(df['revenue'])
```

> **Important:** Do not remove outliers blindly. In fraud detection, outliers *are* the signal.

---

## Step 6 -Standardise Inconsistent Values

String columns often contain multiple representations of the same category -a silent source of errors that corrupts groupings and encodings.

```python
print(df['country'].value_counts().head(20))

df['country'] = (
    df['country']
    .str.lower()
    .str.strip()
    .replace({
        'usa': 'united states',
        'us':  'united states',
        'uk':  'united kingdom',
        'gb':  'united kingdom',
    })
)
```

---

## Step 7 -Encode Categorical Variables

Machine learning models require numerical inputs. The encoding strategy you choose matters and depends on the nature of the category.

```python
# Ordinal encoding -for ordered categories
order_map = {'low': 0, 'medium': 1, 'high': 2}
df['priority'] = df['priority'].map(order_map)

# One-hot encoding -for nominal categories (no inherent order)
df = pd.get_dummies(df, columns=['city'], drop_first=True)
```

> **Watch out for high cardinality.** A column with 500 unique cities will create 499 new columns. Group rare categories into an `'Other'` bucket, or use target encoding.

```python
# Cap rare categories
threshold = 0.01   # categories with < 1% frequency become 'Other'
freq = df['city'].value_counts(normalize=True)
df['city'] = df['city'].where(df['city'].isin(freq[freq >= threshold].index), other='Other')
```

---

## Step 8 -Scale Numerical Features

Many algorithms are sensitive to feature scale: linear models, SVMs, neural networks, and KNN all assume features are on comparable scales. Tree-based models (Random Forest, XGBoost) do not require scaling.

```python
from sklearn.preprocessing import StandardScaler

# StandardScaler: mean=0, std=1 -the default choice for most models
scaler = StandardScaler()
df[['age', 'income']] = scaler.fit_transform(df[['age', 'income']])
```

> **Critical:** Fit the scaler **only on training data**, then use `transform()` on test data. Fitting on the full dataset leaks test information into training.

---

## Step 9 -Split Before You Fit Preprocessors (Prevent Leakage)

This is the most commonly violated rule. **Perform your train/test split before any transformations that require fitting** (imputers, scalers, encoders). Fitting on the full dataset -then splitting -leaks future information into training.

```python
from sklearn.model_selection import train_test_split

X = df.drop('target', axis=1)
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y          # preserves class distribution in both splits
)

# Fit on training data only -then transform both sets
scaler.fit(X_train[numerical_cols])
X_train[numerical_cols] = scaler.transform(X_train[numerical_cols])
X_test[numerical_cols]  = scaler.transform(X_test[numerical_cols])
```

Using `sklearn.pipeline.Pipeline` enforces this pattern by design and is the recommended approach for production code.

---

## Step 10 -Validate the Cleaned Dataset

Before training any model, run a final sanity check. Skipping this step is how silent bugs make it to production.

```python
# No remaining missing values
assert X_train.isnull().sum().sum() == 0, "Missing values remain in training set"
assert X_test.isnull().sum().sum()  == 0, "Missing values remain in test set"

# Correct shapes
print(f"Train: {X_train.shape} | Test: {X_test.shape}")

# Target distribution -check for class imbalance
print(y_train.value_counts(normalize=True).round(3))

# Feature ranges after scaling
print(X_train.describe().round(2))

# Correlation matrix -flag multicollinearity
import seaborn as sns
import matplotlib.pyplot as plt
sns.heatmap(X_train.corr(), cmap='coolwarm', center=0)
plt.title('Feature Correlation Matrix')
plt.show()
```

---

## Quick-Reference Checklist

| # | Step | Done? |
|---|---|:---:|
| 1 | Load and explore -shapes, dtypes, describe | ☐ |
| 2 | Identify and handle missing values | ☐ |
| 3 | Remove duplicate rows | ☐ |
| 4 | Fix incorrect data types and extract datetime features | ☐ |
| 5 | Detect and handle outliers | ☐ |
| 6 | Standardise inconsistent string / category values | ☐ |
| 7 | Encode categorical variables (ordinal / one-hot) | ☐ |
| 8 | Scale numerical features | ☐ |
| 9 | Split *before* fitting any preprocessors | ☐ |
| 10 | Validate -no nulls, correct distributions, no leakage | ☐ |

---

## Final Thought

Data cleaning is not a one-time step -it's an iterative process. You'll discover new issues as you explore deeper, build features, and evaluate models. The best practitioners treat it as a first-class part of the workflow, not a box to tick.

Work through this checklist before every project, and your models will consistently start from a much stronger foundation.
