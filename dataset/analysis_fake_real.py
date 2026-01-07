import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import re
from collections import Counter

# Setup
plt.rcParams['figure.figsize'] = (14, 6)
sns.set_style('whitegrid')
plt.switch_backend('Agg')

# Load final dataset
df = pd.read_csv('final_dataset_for_training.csv')

# Preprocess
df['text'] = df['content'].fillna('').astype(str)
df['word_count'] = df['text'].apply(lambda x: len(x.split()))
df['label_name'] = df['label'].map({0: 'Real', 1: 'Fake'})

print("=" * 60)
print("PHÂN TÍCH FINAL DATASET FOR TRAINING")
print("=" * 60)

# 1. Overview
print("\n1. TỔNG QUAN DATASET")
print("-" * 40)
print(f"Total: {len(df):,} rows")
print(f"  Real (label=0): {(df['label']==0).sum():,} ({(df['label']==0).sum()/len(df)*100:.1f}%)")
print(f"  Fake (label=1): {(df['label']==1).sum():,} ({(df['label']==1).sum()/len(df)*100:.1f}%)")

# 2. Statistics by label
print("\n2. THỐNG KÊ ĐỘ DÀI THEO LABEL")
print("-" * 40)

for label in ['Real', 'Fake']:
    subset = df[df['label_name'] == label]['word_count']
    print(f"\n{label}:")
    print(f"  Count: {len(subset):,}")
    print(f"  Min: {subset.min()} | Max: {subset.max():,}")
    print(f"  Mean: {subset.mean():.1f} | Median: {subset.median():.0f}")
    print(f"  Std: {subset.std():.1f}")
    print(f"  10th percentile: {np.percentile(subset, 10):.0f}")
    print(f"  90th percentile: {np.percentile(subset, 90):.0f}")

# 3. Prepare text for word cloud
print("\n3. CHUẨN BỊ DỮ LIỆU CHO WORD CLOUD")
print("-" * 40)

# Vietnamese stopwords
stopwords_vi = set([
    'và', 'của', 'có', 'là', 'được', 'cho', 'với', 'các', 'trong', 'này',
    'đã', 'không', 'những', 'một', 'để', 'người', 'từ', 'theo', 'khi', 'đến',
    'về', 'như', 'tại', 'còn', 'cũng', 'sẽ', 'nhiều', 'sau', 'trên', 'ra',
    'đó', 'thì', 'nên', 'vì', 'bị', 'hay', 'rất', 'lại', 'nếu', 'đang',
    'nhưng', 'mà', 'vào', 'năm', 'ngày', 'tháng', 'việc', 'làm', 'qua',
    'bạn', 'tôi', 'họ', 'chúng', 'ta', 'mình', 'ai', 'gì', 'nào', 'đây',
    'thế', 'ấy', 'kia', 'đấy', 'sao', 'thôi', 'nhé', 'ạ', 'à', 'ơi',
    'the', 'and', 'to', 'of', 'a', 'in', 'that', 'is', 'for', 'on', 'it',
    'nan', 'none', 'null', 'anh', 'em', 'cô', 'ông', 'bà', 'chị',
    'đi', 'lên', 'xuống', 'vào', 'ra', 'đến', 'về', 'qua', 'lại',
    'rồi', 'nữa', 'thêm', 'hơn', 'nhất', 'quá', 'lắm', 'cả', 'mọi',
    'tất', 'riêng', 'chung', 'khác', 'cùng', 'giữa', 'trước', 'sau',
    'ngoài', 'dưới', 'bên', 'phía', 'đầu', 'cuối', 'giờ', 'lúc'
])

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    words = [w for w in text.split() if w not in stopwords_vi and len(w) > 1]
    return ' '.join(words)

# Get text by label
text_real = ' '.join(df[df['label_name'] == 'Real']['text'].apply(clean_text))
text_fake = ' '.join(df[df['label_name'] == 'Fake']['text'].apply(clean_text))

print("Text prepared for word cloud!")

# 4. Top words analysis
print("\n4. TOP 15 TỪ PHỔ BIẾN")
print("-" * 40)

def get_top_words(text, n=15):
    words = text.split()
    return Counter(words).most_common(n)

print("\nReal News:")
for i, (word, count) in enumerate(get_top_words(text_real, 15), 1):
    print(f"  {i:2}. {word}: {count:,}")

print("\nFake News:")
for i, (word, count) in enumerate(get_top_words(text_fake, 15), 1):
    print(f"  {i:2}. {word}: {count:,}")

# 5. Unique words analysis
print("\n5. TỪ ĐẶC TRƯNG (xuất hiện nhiều ở 1 loại)")
print("-" * 40)

words_real = Counter(text_real.split())
words_fake = Counter(text_fake.split())

# Normalize by total words
total_real = sum(words_real.values())
total_fake = sum(words_fake.values())

# Find words more common in fake
print("\nTừ xuất hiện nhiều hơn trong FAKE (tỷ lệ fake/real):")
fake_dominant = []
for word, count in words_fake.most_common(500):
    if count >= 50:
        real_count = words_real.get(word, 1)
        ratio = (count/total_fake) / (real_count/total_real)
        if ratio > 2:
            fake_dominant.append((word, count, ratio))

fake_dominant.sort(key=lambda x: x[2], reverse=True)
for word, count, ratio in fake_dominant[:10]:
    print(f"  {word}: {count:,} lần (ratio: {ratio:.1f}x)")

print("\nTừ xuất hiện nhiều hơn trong REAL (tỷ lệ real/fake):")
real_dominant = []
for word, count in words_real.most_common(500):
    if count >= 50:
        fake_count = words_fake.get(word, 1)
        ratio = (count/total_real) / (fake_count/total_fake)
        if ratio > 2:
            real_dominant.append((word, count, ratio))

real_dominant.sort(key=lambda x: x[2], reverse=True)
for word, count, ratio in real_dominant[:10]:
    print(f"  {word}: {count:,} lần (ratio: {ratio:.1f}x)")

# ============================================================
# CHARTS
# ============================================================

print("\n" + "=" * 60)
print("VẼ CHARTS...")
print("=" * 60)

# Chart 1: Distribution Bar Chart
fig, ax = plt.subplots(figsize=(8, 5))
labels = ['Real', 'Fake']
counts = [(df['label']==0).sum(), (df['label']==1).sum()]
colors = ['#2ecc71', '#e74c3c']
bars = ax.bar(labels, counts, color=colors)
ax.set_title('Final Dataset: Real vs Fake Distribution', fontsize=14, fontweight='bold')
ax.set_ylabel('Count')
for bar, count in zip(bars, counts):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 100, 
            f'{count:,}\n({count/len(df)*100:.1f}%)', ha='center', fontweight='bold')
plt.tight_layout()
plt.savefig('chart_distribution.png', dpi=150)
plt.close()
print("Saved: chart_distribution.png")

# Chart 2: Word Count Distribution (KDE)
fig, ax = plt.subplots(figsize=(12, 5))
for label, color in [('Real', '#2ecc71'), ('Fake', '#e74c3c')]:
    subset = df[df['label_name'] == label]['word_count']
    subset_clipped = subset[subset <= 1000]
    sns.kdeplot(subset_clipped, ax=ax, label=label, color=color, fill=True, alpha=0.3)
ax.set_title('Word Count Distribution: Real vs Fake', fontsize=14, fontweight='bold')
ax.set_xlabel('Word Count')
ax.set_ylabel('Density')
ax.legend()
ax.set_xlim(0, 1000)
plt.tight_layout()
plt.savefig('chart_word_count_kde.png', dpi=150)
plt.close()
print("Saved: chart_word_count_kde.png")

# Chart 3: Box Plot
fig, ax = plt.subplots(figsize=(8, 5))
df_plot = df[df['word_count'] <= 1000]
sns.boxplot(data=df_plot, x='label_name', y='word_count', ax=ax, 
            palette={'Real': '#2ecc71', 'Fake': '#e74c3c'}, order=['Real', 'Fake'])
ax.set_title('Word Count by Label (Box Plot)', fontsize=14, fontweight='bold')
ax.set_xlabel('Label')
ax.set_ylabel('Word Count')
plt.tight_layout()
plt.savefig('chart_boxplot.png', dpi=150)
plt.close()
print("Saved: chart_boxplot.png")

# Chart 4: Word Clouds
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

wc_params = {
    'width': 800,
    'height': 400,
    'background_color': 'white',
    'max_words': 100,
    'collocations': False
}

# Real
wc1 = WordCloud(**wc_params, colormap='Greens').generate(text_real)
axes[0].imshow(wc1, interpolation='bilinear')
axes[0].set_title('Word Cloud - REAL News', fontsize=14, fontweight='bold', color='green')
axes[0].axis('off')

# Fake
wc2 = WordCloud(**wc_params, colormap='Reds').generate(text_fake)
axes[1].imshow(wc2, interpolation='bilinear')
axes[1].set_title('Word Cloud - FAKE News', fontsize=14, fontweight='bold', color='red')
axes[1].axis('off')

plt.suptitle('Word Cloud Comparison: Real vs Fake', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('chart_wordcloud.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: chart_wordcloud.png")

# Chart 5: Mean/Median Comparison
fig, ax = plt.subplots(figsize=(10, 5))
x = np.arange(2)
width = 0.35

means = [df[df['label']==0]['word_count'].mean(), df[df['label']==1]['word_count'].mean()]
medians = [df[df['label']==0]['word_count'].median(), df[df['label']==1]['word_count'].median()]

bars1 = ax.bar(x - width/2, means, width, label='Mean', color=['#27ae60', '#c0392b'])
bars2 = ax.bar(x + width/2, medians, width, label='Median', color=['#2ecc71', '#e74c3c'], alpha=0.7)

ax.set_title('Word Count Statistics: Real vs Fake', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(['Real', 'Fake'])
ax.set_ylabel('Word Count')
ax.legend()

for bar, val in zip(bars1, means):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5, f'{val:.0f}', ha='center', fontsize=10)
for bar, val in zip(bars2, medians):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5, f'{val:.0f}', ha='center', fontsize=10)

plt.tight_layout()
plt.savefig('chart_stats_comparison.png', dpi=150)
plt.close()
print("Saved: chart_stats_comparison.png")

# Chart 6: Percentile Distribution
fig, ax = plt.subplots(figsize=(12, 5))
percentiles = np.arange(0, 101, 5)
real_pct = [np.percentile(df[df['label']==0]['word_count'], p) for p in percentiles]
fake_pct = [np.percentile(df[df['label']==1]['word_count'], p) for p in percentiles]

ax.plot(percentiles, real_pct, 'g-', linewidth=2, label='Real', marker='o', markersize=4)
ax.plot(percentiles, fake_pct, 'r-', linewidth=2, label='Fake', marker='s', markersize=4)
ax.fill_between(percentiles, real_pct, alpha=0.2, color='green')
ax.fill_between(percentiles, fake_pct, alpha=0.2, color='red')

ax.set_title('Word Count by Percentile: Real vs Fake', fontsize=14, fontweight='bold')
ax.set_xlabel('Percentile')
ax.set_ylabel('Word Count')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_xlim(0, 100)

plt.tight_layout()
plt.savefig('chart_percentile.png', dpi=150)
plt.close()
print("Saved: chart_percentile.png")

print("\n" + "=" * 60)
print("HOÀN THÀNH!")
print("=" * 60)
