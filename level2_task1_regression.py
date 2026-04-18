"""
Level 2 - Task 1: Regression Analysis
Dataset: House Prediction (Boston-style dataset)
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Load with proper column names (Boston Housing format)
cols = ['CRIM','ZN','INDUS','CHAS','NOX','RM','AGE','DIS','RAD','TAX','PTRATIO','B','LSTAT','MEDV']
df = pd.read_csv('/mnt/user-data/uploads/4__house_Prediction_Data_Set.csv',
                 sep=r'\s+', header=None, names=cols)

print("=" * 60)
print("LEVEL 2 – TASK 1: REGRESSION ANALYSIS")
print("Dataset: Boston Housing")
print("=" * 60)
print(f"\n▶ Shape: {df.shape}")
print(f"\n▶ Preview:\n{df.head()}")
print(f"\n▶ Describe:\n{df.describe().round(3)}")
print(f"\n▶ Missing values: {df.isnull().sum().sum()}")

# ── Feature & Target ──────────────────────────────────────────────────────────
# Predict MEDV (Median House Value) using all other features
X = df.drop('MEDV', axis=1)
y = df['MEDV']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"\n▶ Train size: {len(X_train)}  |  Test size: {len(X_test)}")

# Scale features
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

# ── Linear Regression ─────────────────────────────────────────────────────────
model = LinearRegression()
model.fit(X_train_s, y_train)
y_pred = model.predict(X_test_s)

r2  = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mse)

print("\n▶ Model Performance:")
print(f"   R² Score  : {r2:.4f}")
print(f"   MSE       : {mse:.4f}")
print(f"   RMSE      : {rmse:.4f}")
print(f"   MAE       : {mae:.4f}")

print("\n▶ Feature Coefficients:")
coef_df = pd.DataFrame({'Feature': X.columns, 'Coefficient': model.coef_})
coef_df = coef_df.sort_values('Coefficient')
print(coef_df.to_string(index=False))

# ── Visualisations ────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(13, 10))

# 1. Actual vs Predicted
ax = axes[0, 0]
ax.scatter(y_test, y_pred, alpha=0.6, color='#4C72B0', edgecolors='white', s=50)
lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
ax.plot(lims, lims, 'r--', linewidth=1.5, label='Perfect Fit')
ax.set_xlabel('Actual MEDV ($000s)', fontsize=10); ax.set_ylabel('Predicted MEDV ($000s)', fontsize=10)
ax.set_title(f'Actual vs Predicted\nR² = {r2:.4f}', fontsize=11, fontweight='bold')
ax.legend(); ax.grid(alpha=0.3)

# 2. Residuals Distribution
ax = axes[0, 1]
residuals = y_test - y_pred
ax.hist(residuals, bins=25, color='#DD8452', edgecolor='white', alpha=0.85)
ax.axvline(0, color='red', linestyle='--', linewidth=1.5)
ax.set_xlabel('Residual', fontsize=10); ax.set_ylabel('Count', fontsize=10)
ax.set_title('Residuals Distribution', fontsize=11, fontweight='bold')
ax.grid(alpha=0.3)

# 3. Residuals vs Predicted
ax = axes[1, 0]
ax.scatter(y_pred, residuals, alpha=0.6, color='#55A868', edgecolors='white', s=50)
ax.axhline(0, color='red', linestyle='--', linewidth=1.5)
ax.set_xlabel('Predicted Values', fontsize=10); ax.set_ylabel('Residuals', fontsize=10)
ax.set_title('Residuals vs Predicted (Homoscedasticity Check)', fontsize=11, fontweight='bold')
ax.grid(alpha=0.3)

# 4. Feature Importance (Coefficients)
ax = axes[1, 1]
colors_bar = ['#C44E52' if c < 0 else '#4C72B0' for c in coef_df['Coefficient']]
ax.barh(coef_df['Feature'], coef_df['Coefficient'], color=colors_bar, edgecolor='white')
ax.axvline(0, color='black', linewidth=0.8)
ax.set_xlabel('Coefficient (Standardised)', fontsize=10)
ax.set_title('Feature Coefficients', fontsize=11, fontweight='bold')
ax.grid(axis='x', alpha=0.3)

plt.suptitle('Level 2 – Task 1: Linear Regression Analysis\n(Predicting House Median Value)',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('/home/claude/L2T1_regression.png', dpi=150, bbox_inches='tight')
plt.close()

print("\n✅ Regression analysis complete – plot saved to L2T1_regression.png")
