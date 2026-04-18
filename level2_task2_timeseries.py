"""
Level 2 - Task 2: Time Series Analysis
Dataset: Stock Prices
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('/mnt/user-data/uploads/2__Stock_Prices_Data_Set.csv')
df['date'] = pd.to_datetime(df['date'])

print("=" * 60)
print("LEVEL 2 – TASK 2: TIME SERIES ANALYSIS")
print("Dataset: Stock Prices")
print("=" * 60)
print(f"\n▶ Shape: {df.shape}")
print(f"▶ Date range: {df['date'].min().date()} → {df['date'].max().date()}")
print(f"▶ Symbols: {df['symbol'].nunique()}")
print(f"▶ Top symbols: {df['symbol'].value_counts().head(5).to_dict()}")

# ── Focus on AAPL ─────────────────────────────────────────────────────────────
aapl = df[df['symbol'] == 'AAPL'].sort_values('date').copy()
aapl.set_index('date', inplace=True)
print(f"\n▶ AAPL rows: {len(aapl)}")
print(f"▶ Missing values: {aapl['close'].isnull().sum()}")

# ── Moving Averages ───────────────────────────────────────────────────────────
aapl['MA_7']  = aapl['close'].rolling(window=7).mean()
aapl['MA_30'] = aapl['close'].rolling(window=30).mean()
aapl['MA_90'] = aapl['close'].rolling(window=90).mean()

# ── Manual Decomposition (trend, residuals) ───────────────────────────────────
# Trend: 30-day rolling mean
aapl['trend']    = aapl['close'].rolling(window=30, center=True).mean()
aapl['residual'] = aapl['close'] - aapl['trend']
# Seasonality approximation: mean residual by day-of-week
aapl['dow'] = aapl.index.dayofweek
seasonal_pattern = aapl.groupby('dow')['residual'].mean()
aapl['seasonal'] = aapl['dow'].map(seasonal_pattern)
aapl['irregular'] = aapl['residual'] - aapl['seasonal']

# ── Volatility (rolling std) ──────────────────────────────────────────────────
aapl['volatility_30'] = aapl['close'].rolling(30).std()

print(f"\n▶ AAPL Closing Price Stats:")
print(aapl['close'].describe().round(3))

# ── Figure 1: Price + Moving Averages ────────────────────────────────────────
fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)

ax = axes[0]
ax.plot(aapl.index, aapl['close'], color='#4C72B0', linewidth=1, label='Close', alpha=0.7)
ax.plot(aapl.index, aapl['MA_7'],  color='#DD8452', linewidth=1.5, label='7-Day MA')
ax.plot(aapl.index, aapl['MA_30'], color='#55A868', linewidth=2,   label='30-Day MA')
ax.plot(aapl.index, aapl['MA_90'], color='#C44E52', linewidth=2,   label='90-Day MA')
ax.set_ylabel('Price ($)', fontsize=10); ax.set_title('AAPL Closing Price + Moving Averages', fontsize=11, fontweight='bold')
ax.legend(fontsize=9); ax.grid(alpha=0.3)

ax = axes[1]
ax.fill_between(aapl.index, aapl['low'], aapl['high'], alpha=0.2, color='#4C72B0', label='High-Low Range')
ax.plot(aapl.index, aapl['close'], color='#4C72B0', linewidth=1)
ax.set_ylabel('Price ($)', fontsize=10); ax.set_title('AAPL Daily High-Low Range', fontsize=11, fontweight='bold')
ax.legend(fontsize=9); ax.grid(alpha=0.3)

ax = axes[2]
ax.bar(aapl.index, aapl['volume'], color='#9575CD', alpha=0.5, label='Volume')
ax.set_ylabel('Volume', fontsize=10); ax.set_title('Trading Volume', fontsize=11, fontweight='bold')
ax.legend(fontsize=9); ax.grid(alpha=0.3)
ax.set_xlabel('Date', fontsize=10)

plt.suptitle("Level 2 – Task 2: AAPL Time Series Analysis", fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('/home/claude/L2T2_timeseries.png', dpi=150, bbox_inches='tight')
plt.close()

# ── Figure 2: Manual Decomposition ───────────────────────────────────────────
fig, axes = plt.subplots(4, 1, figsize=(14, 14), sharex=True)
components = [('close', 'Observed', '#4C72B0'), ('trend', 'Trend (30-Day MA)', '#DD8452'),
              ('seasonal', 'Seasonality (Day-of-Week)', '#55A868'), ('irregular', 'Irregular / Residual', '#C44E52')]
for (col, title, c), ax in zip(components, axes):
    if col == 'seasonal':
        ax.bar(aapl.index, aapl[col], color=c, alpha=0.6)
    else:
        ax.plot(aapl.index, aapl[col], color=c, linewidth=1.2)
    ax.set_ylabel(title, fontsize=9); ax.set_title(title, fontsize=10, fontweight='bold')
    ax.axhline(0, color='black', linewidth=0.5, linestyle='--')
    ax.grid(alpha=0.3)
axes[-1].set_xlabel('Date', fontsize=10)
plt.suptitle("Level 2 – Task 2: Time Series Decomposition (AAPL)", fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('/home/claude/L2T2_decomposition.png', dpi=150, bbox_inches='tight')
plt.close()

print("\n✅ Time series analysis complete – 2 plots saved")
