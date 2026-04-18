"""
Level 1 - Task 2: Exploratory Data Analysis (EDA)
Dataset: Iris
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('/mnt/user-data/uploads/1__iris.csv')
numeric = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']

print("=" * 60)
print("LEVEL 1 – TASK 2: EXPLORATORY DATA ANALYSIS (EDA)")
print("=" * 60)

# ── Summary Statistics ────────────────────────────────────────────────────────
print("\n▶ Summary Statistics:")
stats = df[numeric].agg(['mean', 'median', 'std', 'min', 'max'])
stats.loc['mode'] = [df[c].mode()[0] for c in numeric]
print(stats.round(3))

# ── Correlation Matrix ────────────────────────────────────────────────────────
corr = df[numeric].corr()
print("\n▶ Correlation Matrix:")
print(corr.round(3))

# ── Figure 1: Histograms ──────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
colors = ['#4C72B0', '#DD8452', '#55A868', '#C44E52']
for i, (col, ax, c) in enumerate(zip(numeric, axes.flat, colors)):
    ax.hist(df[col], bins=20, color=c, edgecolor='white', alpha=0.85)
    ax.axvline(df[col].mean(), color='black', linestyle='--', linewidth=1.5, label=f'Mean={df[col].mean():.2f}')
    ax.axvline(df[col].median(), color='red', linestyle=':', linewidth=1.5, label=f'Median={df[col].median():.2f}')
    ax.set_title(col.replace('_', ' ').title(), fontsize=11, fontweight='bold')
    ax.set_xlabel('Value'); ax.set_ylabel('Frequency')
    ax.legend(fontsize=8)
plt.suptitle("Level 1 – Task 2: Feature Distributions (Histograms)", fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('/home/claude/L1T2_histograms.png', dpi=150, bbox_inches='tight')
plt.close()

# ── Figure 2: Boxplots by Species ─────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
palette = {'Setosa': '#4C72B0', 'Versicolor': '#DD8452', 'Virginica': '#55A868'}
df['species'] = df['species'].str.strip().str.lower().str.capitalize()
for col, ax in zip(numeric, axes.flat):
    sns.boxplot(data=df, x='species', y=col, palette=palette, ax=ax)
    ax.set_title(col.replace('_', ' ').title(), fontsize=11, fontweight='bold')
    ax.set_xlabel('Species'); ax.set_ylabel(col)
plt.suptitle("Level 1 – Task 2: Boxplots by Species", fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('/home/claude/L1T2_boxplots.png', dpi=150, bbox_inches='tight')
plt.close()

# ── Figure 3: Scatter plots (pairplot-style) ──────────────────────────────────
pairs = [('sepal_length', 'sepal_width'), ('sepal_length', 'petal_length'),
         ('petal_length', 'petal_width'), ('sepal_width', 'petal_width')]
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
species_list = df['species'].unique()
colors2 = ['#4C72B0', '#DD8452', '#55A868']
for (x, y), ax in zip(pairs, axes.flat):
    for sp, c in zip(species_list, colors2):
        sub = df[df['species'] == sp]
        ax.scatter(sub[x], sub[y], c=c, label=sp, alpha=0.7, edgecolors='white', s=50)
    ax.set_xlabel(x.replace('_', ' ').title(), fontsize=9)
    ax.set_ylabel(y.replace('_', ' ').title(), fontsize=9)
    ax.set_title(f'{x} vs {y}', fontsize=10, fontweight='bold')
    ax.legend(fontsize=8)
plt.suptitle("Level 1 – Task 2: Scatter Plots by Species", fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('/home/claude/L1T2_scatter.png', dpi=150, bbox_inches='tight')
plt.close()

# ── Figure 4: Correlation Heatmap ─────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 5))
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, annot=True, fmt='.3f', cmap='coolwarm', mask=mask,
            linewidths=0.5, ax=ax, vmin=-1, vmax=1, square=True)
ax.set_title("Correlation Heatmap – Iris Features", fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('/home/claude/L1T2_correlation.png', dpi=150, bbox_inches='tight')
plt.close()

print("\n✅ All EDA plots saved (histograms, boxplots, scatter, correlation heatmap)")
