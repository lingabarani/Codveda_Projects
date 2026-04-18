"""
Level 1 - Task 1: Data Cleaning and Preprocessing
Dataset: Iris (iris.csv)
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# ── Load ──────────────────────────────────────────────────────────────────────
df = pd.read_csv('/mnt/user-data/uploads/1__iris.csv')
print("=" * 60)
print("LEVEL 1 – TASK 1: DATA CLEANING AND PREPROCESSING")
print("=" * 60)
print(f"\n▶ Original shape: {df.shape}")
print(f"\n▶ First 5 rows:\n{df.head()}")
print(f"\n▶ Data Types:\n{df.dtypes}")

# ── Missing Values ────────────────────────────────────────────────────────────
print("\n▶ Missing values per column:")
print(df.isnull().sum())

# Artificially inject missing values for demonstration (10% random nulls)
np.random.seed(42)
for col in ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']:
    idx = np.random.choice(df.index, size=int(0.10 * len(df)), replace=False)
    df.loc[idx, col] = np.nan

print(f"\n▶ After injecting 10% nulls:")
print(df.isnull().sum())

# Impute numerical columns with median
for col in ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']:
    median_val = df[col].median()
    df[col].fillna(median_val, inplace=True)
    print(f"   Imputed '{col}' missing values with median = {median_val:.3f}")

print(f"\n▶ Missing values after imputation:\n{df.isnull().sum()}")

# ── Duplicates ────────────────────────────────────────────────────────────────
dup_count = df.duplicated().sum()
print(f"\n▶ Duplicate rows found: {dup_count}")
df.drop_duplicates(inplace=True)
print(f"▶ Shape after removing duplicates: {df.shape}")

# ── Standardise categorical column ───────────────────────────────────────────
print(f"\n▶ Unique species (before standardisation): {df['species'].unique()}")
df['species'] = df['species'].str.strip().str.lower().str.capitalize()
print(f"▶ Unique species (after standardisation):  {df['species'].unique()}")

# ── Summary after cleaning ────────────────────────────────────────────────────
print(f"\n▶ Final cleaned dataset shape: {df.shape}")
print(f"\n▶ Statistical summary:\n{df.describe().round(3)}")

# ── Visualise missing-value heatmap (before / after) ─────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(10, 4))

# Re-create a "dirty" snapshot for visualisation
dirty = df.copy()
for col in ['sepal_length', 'sepal_width']:
    idx = np.random.choice(dirty.index, size=5, replace=False)
    dirty.loc[idx, col] = np.nan

axes[0].set_title("Before Cleaning\n(null pattern illustration)", fontsize=11, fontweight='bold')
axes[0].imshow(dirty.isnull().astype(int).values, aspect='auto', cmap='Reds', interpolation='nearest')
axes[0].set_xlabel("Columns"); axes[0].set_ylabel("Rows")
axes[0].set_xticks(range(len(dirty.columns))); axes[0].set_xticklabels(dirty.columns, rotation=45, ha='right', fontsize=8)

axes[1].set_title("After Cleaning\n(no nulls)", fontsize=11, fontweight='bold')
axes[1].imshow(df.isnull().astype(int).values, aspect='auto', cmap='Greens', interpolation='nearest')
axes[1].set_xlabel("Columns"); axes[1].set_ylabel("Rows")
axes[1].set_xticks(range(len(df.columns))); axes[1].set_xticklabels(df.columns, rotation=45, ha='right', fontsize=8)

plt.suptitle("Level 1 – Task 1: Missing Value Heatmap", fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('/home/claude/L1T1_cleaning.png', dpi=150, bbox_inches='tight')
plt.close()

# Save cleaned CSV
df.to_csv('/home/claude/iris_cleaned.csv', index=False)
print("\n✅ Cleaned dataset saved to iris_cleaned.csv")
print("✅ Plot saved to L1T1_cleaning.png")
