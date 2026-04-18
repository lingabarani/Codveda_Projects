"""
Level 1 - Task 3: Basic Data Visualization
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
df['species'] = df['species'].str.strip().str.lower().str.capitalize()
numeric = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']

print("=" * 60)
print("LEVEL 1 – TASK 3: BASIC DATA VISUALIZATION")
print("=" * 60)

# ── Bar Plot: Mean feature values per species ─────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 6))
means = df.groupby('species')[numeric].mean()
x = np.arange(len(numeric))
width = 0.25
colors = ['#4C72B0', '#DD8452', '#55A868']
for i, (sp, c) in enumerate(zip(means.index, colors)):
    bars = ax.bar(x + i*width, means.loc[sp], width, label=sp, color=c, edgecolor='white')
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                f'{bar.get_height():.2f}', ha='center', va='bottom', fontsize=7)
ax.set_xticks(x + width)
ax.set_xticklabels([c.replace('_', '\n').title() for c in numeric], fontsize=10)
ax.set_xlabel('Feature', fontsize=12)
ax.set_ylabel('Mean Value (cm)', fontsize=12)
ax.set_title('Mean Feature Values per Species – Iris Dataset', fontsize=13, fontweight='bold')
ax.legend(title='Species', fontsize=10)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('/home/claude/L1T3_barplot.png', dpi=150, bbox_inches='tight')
plt.close()
print("✅ Bar plot saved")

# ── Line Chart: Sorted sepal_length to show distribution trend ────────────────
fig, ax = plt.subplots(figsize=(12, 5))
for sp, c in zip(df['species'].unique(), colors):
    sub = df[df['species'] == sp]['sepal_length'].sort_values().reset_index(drop=True)
    ax.plot(sub.index, sub.values, label=sp, color=c, linewidth=2, marker='o', markersize=3, alpha=0.8)
ax.set_xlabel('Sorted Sample Index', fontsize=12)
ax.set_ylabel('Sepal Length (cm)', fontsize=12)
ax.set_title('Sepal Length Distribution Trend by Species (Sorted)', fontsize=13, fontweight='bold')
ax.legend(title='Species', fontsize=10)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('/home/claude/L1T3_linechart.png', dpi=150, bbox_inches='tight')
plt.close()
print("✅ Line chart saved")

# ── Scatter Plot: Petal Length vs Petal Width ─────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 6))
for sp, c in zip(df['species'].unique(), colors):
    sub = df[df['species'] == sp]
    ax.scatter(sub['petal_length'], sub['petal_width'], c=c, label=sp,
               s=60, edgecolors='white', linewidth=0.5, alpha=0.85)
# Trend line
z = np.polyfit(df['petal_length'], df['petal_width'], 1)
p = np.poly1d(z)
x_line = np.linspace(df['petal_length'].min(), df['petal_length'].max(), 100)
ax.plot(x_line, p(x_line), 'k--', linewidth=1.5, label='Trend Line', alpha=0.6)
ax.set_xlabel('Petal Length (cm)', fontsize=12)
ax.set_ylabel('Petal Width (cm)', fontsize=12)
ax.set_title('Petal Length vs Petal Width – Iris Dataset', fontsize=13, fontweight='bold')
ax.legend(title='Species', fontsize=10)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('/home/claude/L1T3_scatter.png', dpi=150, bbox_inches='tight')
plt.close()
print("✅ Scatter plot saved")

print("\n✅ Level 1 Task 3 complete – all 3 plots exported")
