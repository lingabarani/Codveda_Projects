"""
Level 3 - Task 1: Predictive Modeling – Classification (Customer Churn)
Datasets: churn-bigml-80.csv (train) + churn-bigml-20.csv (test)
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, confusion_matrix, classification_report,
                              roc_auc_score, roc_curve)
import warnings
warnings.filterwarnings('ignore')

# ── Load Data ─────────────────────────────────────────────────────────────────
train = pd.read_csv('/mnt/user-data/uploads/churn-bigml-80.csv')
test  = pd.read_csv('/mnt/user-data/uploads/churn-bigml-20.csv')

print("=" * 60)
print("LEVEL 3 – TASK 1: PREDICTIVE MODELING (CLASSIFICATION)")
print("Dataset: Telecom Customer Churn")
print("=" * 60)
print(f"\n▶ Train shape: {train.shape}  |  Test shape: {test.shape}")
print(f"\n▶ Churn distribution (train):\n{train['Churn'].value_counts()}")
print(f"\n▶ Missing values (train): {train.isnull().sum().sum()}")

# ── Preprocessing ─────────────────────────────────────────────────────────────
le = LabelEncoder()
for df in [train, test]:
    df['State']               = le.fit_transform(df['State'])
    df['International plan']  = (df['International plan'] == 'Yes').astype(int)
    df['Voice mail plan']     = (df['Voice mail plan'] == 'Yes').astype(int)
    df['Churn']               = (df['Churn'] == True).astype(int)

feature_cols = [c for c in train.columns if c not in ['Churn', 'Area code']]
X_train = train[feature_cols]; y_train = train['Churn']
X_test  = test[feature_cols];  y_test  = test['Churn']

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

print(f"\n▶ Features used: {len(feature_cols)}")
print(f"▶ Class balance (train): {y_train.value_counts().to_dict()}")

# ── Train 3 Models ────────────────────────────────────────────────────────────
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Decision Tree':       DecisionTreeClassifier(random_state=42),
    'Random Forest':       RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
}
results = {}
for name, model in models.items():
    model.fit(X_train_s, y_train)
    y_pred = model.predict(X_test_s)
    y_prob = model.predict_proba(X_test_s)[:, 1]
    results[name] = {
        'model': model,
        'y_pred': y_pred,
        'y_prob': y_prob,
        'accuracy':  accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall':    recall_score(y_test, y_pred),
        'f1':        f1_score(y_test, y_pred),
        'auc':       roc_auc_score(y_test, y_prob),
    }

print("\n▶ Model Comparison:")
comparison = pd.DataFrame({
    name: {k: v for k, v in r.items() if k not in ['model', 'y_pred', 'y_prob']}
    for name, r in results.items()
}).T.round(4)
print(comparison)

# ── Grid Search on Random Forest ─────────────────────────────────────────────
param_grid = {'n_estimators': [50, 100], 'max_depth': [5, 10, None], 'min_samples_split': [2, 5]}
gs = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=3,
                  scoring='f1', n_jobs=-1, verbose=0)
gs.fit(X_train_s, y_train)
best_rf = gs.best_estimator_
y_pred_best = best_rf.predict(X_test_s)
y_prob_best = best_rf.predict_proba(X_test_s)[:, 1]
print(f"\n▶ Best RF Params (GridSearch): {gs.best_params_}")
print(f"▶ Best RF F1 (CV): {gs.best_score_:.4f}")
print(f"\n▶ Tuned RF Classification Report:\n{classification_report(y_test, y_pred_best, target_names=['No Churn','Churn'])}")

# ── Feature Importance ────────────────────────────────────────────────────────
feat_imp = pd.Series(best_rf.feature_importances_, index=feature_cols).sort_values(ascending=False)

# ── Visualisations ────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(16, 11))

# 1-3. Confusion Matrices
for i, (name, r) in enumerate(results.items()):
    ax = axes[0, i]
    cm = confusion_matrix(y_test, r['y_pred'])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['No Churn','Churn'], yticklabels=['No Churn','Churn'])
    ax.set_title(f'{name}\nAcc={r["accuracy"]:.3f} F1={r["f1"]:.3f}', fontsize=10, fontweight='bold')
    ax.set_xlabel('Predicted'); ax.set_ylabel('Actual')

# 4. ROC Curves
ax = axes[1, 0]
for name, r in results.items():
    fpr, tpr, _ = roc_curve(y_test, r['y_prob'])
    ax.plot(fpr, tpr, linewidth=2, label=f"{name} (AUC={r['auc']:.3f})")
ax.plot([0,1],[0,1],'k--',linewidth=1)
ax.set_xlabel('False Positive Rate', fontsize=10); ax.set_ylabel('True Positive Rate', fontsize=10)
ax.set_title('ROC Curves', fontsize=11, fontweight='bold')
ax.legend(fontsize=8); ax.grid(alpha=0.3)

# 5. Metric Comparison Bar Chart
ax = axes[1, 1]
metrics = ['accuracy','precision','recall','f1','auc']
x = np.arange(len(metrics)); width = 0.25
bar_colors = ['#4C72B0','#DD8452','#55A868']
for i, (name, r) in enumerate(results.items()):
    vals = [r[m] for m in metrics]
    ax.bar(x + i*width, vals, width, label=name, color=bar_colors[i], edgecolor='white')
ax.set_xticks(x + width); ax.set_xticklabels(metrics, fontsize=9)
ax.set_ylim(0, 1.1); ax.set_ylabel('Score', fontsize=10)
ax.set_title('Model Metrics Comparison', fontsize=11, fontweight='bold')
ax.legend(fontsize=8); ax.grid(axis='y', alpha=0.3)

# 6. Feature Importance
ax = axes[1, 2]
top15 = feat_imp.head(15)
ax.barh(top15.index[::-1], top15.values[::-1], color='#4C72B0', edgecolor='white')
ax.set_xlabel('Importance', fontsize=10)
ax.set_title('Top 15 Feature Importances\n(Tuned Random Forest)', fontsize=11, fontweight='bold')
ax.grid(axis='x', alpha=0.3)

plt.suptitle("Level 3 – Task 1: Customer Churn Prediction – Classification Models",
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('/home/claude/L3T1_classification.png', dpi=150, bbox_inches='tight')
plt.close()

print("\n✅ Classification analysis complete – plot saved to L3T1_classification.png")
