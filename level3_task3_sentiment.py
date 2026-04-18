"""
Level 3 - Task 3: NLP – Sentiment Analysis
Dataset: Sentiment_dataset.csv
Uses pure Python + regex + scipy (no nltk/textblob/wordcloud needed)
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import re
import string
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('/mnt/user-data/uploads/3__Sentiment_dataset.csv')

print("=" * 60)
print("LEVEL 3 – TASK 3: NLP – SENTIMENT ANALYSIS")
print("Dataset: Social Media Sentiment")
print("=" * 60)
print(f"\n▶ Shape: {df.shape}")
print(f"▶ Columns: {list(df.columns)}")
print(f"\n▶ First 3 rows:\n{df[['Text','Sentiment','Platform','Country']].head(3)}")
print(f"\n▶ Sentiment distribution:\n{df['Sentiment'].str.strip().value_counts()}")

# ── Text Preprocessing ────────────────────────────────────────────────────────
STOPWORDS = set([
    'i','me','my','myself','we','our','ours','ourselves','you','your','yours',
    'yourself','he','him','his','himself','she','her','hers','herself','it','its',
    'itself','they','them','their','theirs','themselves','what','which','who',
    'whom','this','that','these','those','am','is','are','was','were','be','been',
    'being','have','has','had','having','do','does','did','doing','a','an','the',
    'and','but','if','or','because','as','until','while','of','at','by','for',
    'with','about','against','between','into','through','during','before','after',
    'above','below','to','from','up','down','in','out','on','off','over','under',
    'again','further','then','once','here','there','when','where','why','how',
    'all','both','each','few','more','most','other','some','such','no','nor',
    'not','only','own','same','so','than','too','very','s','t','just','don',
    'should','now','d','ll','m','o','re','ve','y','ain','aren','couldn','didn',
    'doesn','hadn','hasn','haven','isn','ma','mightn','mustn','needn','shan',
    'shouldn','wasn','weren','won','wouldn','got','get','go','going','went',
    'day','today','time','like','also','one','two','three','new','good','will',
    'really','still','even','back','much','many','made','make','came','come',
])

def preprocess_text(text):
    """Tokenise, lowercase, remove punctuation/stopwords, and stem (simple)."""
    if not isinstance(text, str):
        return []
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', '', text)          # remove URLs
    text = re.sub(r'@\w+|#\w+', '', text)               # remove mentions/hashtags
    text = re.sub(r'[^\w\s]', ' ', text)                 # remove punctuation
    text = re.sub(r'\d+', '', text)                      # remove digits
    tokens = text.split()
    tokens = [t for t in tokens if t not in STOPWORDS and len(t) > 2]
    return tokens

df['Sentiment_clean'] = df['Sentiment'].str.strip()
df['Tokens']          = df['Text'].apply(preprocess_text)
df['Token_count']     = df['Tokens'].apply(len)
df['Text_length']     = df['Text'].apply(lambda x: len(str(x)))

print(f"\n▶ Average token count per sentiment:")
print(df.groupby('Sentiment_clean')['Token_count'].mean().round(2))

# ── Word frequency per sentiment ──────────────────────────────────────────────
sentiment_words = {}
for sent in df['Sentiment_clean'].unique():
    all_tokens = [t for tokens in df[df['Sentiment_clean'] == sent]['Tokens'] for t in tokens]
    sentiment_words[sent] = Counter(all_tokens)

print("\n▶ Top 10 words per sentiment:")
for sent, counter in sentiment_words.items():
    print(f"   {sent}: {[w for w, _ in counter.most_common(10)]}")

# ── Platform & Country distribution ──────────────────────────────────────────
df['Platform_clean'] = df['Platform'].str.strip()
df['Country_clean']  = df['Country'].str.strip()

# ── Figure 1: Sentiment overview ──────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(16, 10))

# 1. Sentiment pie
ax = axes[0, 0]
counts = df['Sentiment_clean'].value_counts()
colors_pie = {'Positive': '#55A868', 'Negative': '#C44E52', 'Neutral': '#8C8C8C'}
c_list = [colors_pie.get(s, '#4C72B0') for s in counts.index]
wedges, texts, autotexts = ax.pie(counts.values, labels=counts.index, autopct='%1.1f%%',
                                   colors=c_list, startangle=140, pctdistance=0.8)
for at in autotexts: at.set_fontsize(10)
ax.set_title('Sentiment Distribution', fontsize=11, fontweight='bold')

# 2. Sentiment by Platform
ax = axes[0, 1]
plat_sent = df.groupby(['Platform_clean', 'Sentiment_clean']).size().unstack(fill_value=0)
plat_sent.plot(kind='bar', ax=ax, color=[colors_pie.get(c,'#4C72B0') for c in plat_sent.columns],
               edgecolor='white')
ax.set_xlabel('Platform', fontsize=10); ax.set_ylabel('Count', fontsize=10)
ax.set_title('Sentiment by Platform', fontsize=11, fontweight='bold')
ax.legend(fontsize=8); ax.tick_params(axis='x', rotation=20)
ax.grid(axis='y', alpha=0.3)

# 3. Sentiment by Country
ax = axes[0, 2]
country_sent = df.groupby(['Country_clean', 'Sentiment_clean']).size().unstack(fill_value=0)
country_sent.plot(kind='bar', ax=ax, color=[colors_pie.get(c,'#4C72B0') for c in country_sent.columns],
                  edgecolor='white')
ax.set_xlabel('Country', fontsize=10); ax.set_ylabel('Count', fontsize=10)
ax.set_title('Sentiment by Country', fontsize=11, fontweight='bold')
ax.legend(fontsize=8); ax.tick_params(axis='x', rotation=30)
ax.grid(axis='y', alpha=0.3)

# 4. Token count distribution
ax = axes[1, 0]
for sent, c in colors_pie.items():
    sub = df[df['Sentiment_clean'] == sent]['Token_count']
    if len(sub): ax.hist(sub, bins=20, alpha=0.6, label=sent, color=c, edgecolor='white')
ax.set_xlabel('Token Count', fontsize=10); ax.set_ylabel('Frequency', fontsize=10)
ax.set_title('Token Count Distribution by Sentiment', fontsize=11, fontweight='bold')
ax.legend(fontsize=9); ax.grid(alpha=0.3)

# 5. Top words – Positive
ax = axes[1, 1]
pos_top = pd.Series(dict(sentiment_words.get('Positive', Counter()).most_common(15)))
ax.barh(pos_top.index[::-1], pos_top.values[::-1], color='#55A868', edgecolor='white')
ax.set_title('Top 15 Words – Positive', fontsize=11, fontweight='bold')
ax.set_xlabel('Frequency'); ax.grid(axis='x', alpha=0.3)

# 6. Top words – Negative
ax = axes[1, 2]
neg_top = pd.Series(dict(sentiment_words.get('Negative', Counter()).most_common(15)))
ax.barh(neg_top.index[::-1], neg_top.values[::-1], color='#C44E52', edgecolor='white')
ax.set_title('Top 15 Words – Negative', fontsize=11, fontweight='bold')
ax.set_xlabel('Frequency'); ax.grid(axis='x', alpha=0.3)

plt.suptitle("Level 3 – Task 3: NLP Sentiment Analysis – Social Media Dataset",
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('/home/claude/L3T3_sentiment.png', dpi=150, bbox_inches='tight')
plt.close()

# ── Figure 2: Word-frequency "cloud" as styled bar chart ──────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 6))
for ax, sent, color in zip(axes, ['Positive','Negative','Neutral'],
                            ['#55A868','#C44E52','#8C8C8C']):
    top = pd.Series(dict(sentiment_words.get(sent, Counter()).most_common(20)))
    if len(top):
        font_sizes = np.interp(top.values, (top.values.min(), top.values.max()), (10, 28))
        ax.barh(top.index[::-1], top.values[::-1], color=color, alpha=0.8, edgecolor='white')
        ax.set_title(f'{sent}\nTop 20 Words', fontsize=12, fontweight='bold',
                     color=color if sent != 'Neutral' else '#555')
        ax.set_xlabel('Frequency', fontsize=9)
        ax.grid(axis='x', alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'No data', ha='center', va='center')

plt.suptitle("Level 3 – Task 3: Word Frequency by Sentiment Class",
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('/home/claude/L3T3_wordfreq.png', dpi=150, bbox_inches='tight')
plt.close()

# ── Figure 3: Monthly sentiment trend ─────────────────────────────────────────
if 'Year' in df.columns and 'Month' in df.columns:
    df['Period'] = pd.to_datetime(df[['Year','Month']].assign(Day=1))
    trend = df.groupby(['Period','Sentiment_clean']).size().unstack(fill_value=0)
    fig, ax = plt.subplots(figsize=(12, 5))
    for sent in trend.columns:
        ax.plot(trend.index, trend[sent], marker='o', label=sent,
                color=colors_pie.get(sent,'#4C72B0'), linewidth=2, markersize=5)
    ax.set_xlabel('Month', fontsize=10); ax.set_ylabel('Post Count', fontsize=10)
    ax.set_title('Monthly Sentiment Trend Over Time', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('/home/claude/L3T3_trend.png', dpi=150, bbox_inches='tight')
    plt.close()

print("\n✅ Sentiment analysis complete – 3 plots saved")
