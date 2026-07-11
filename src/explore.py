import os
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # save figures without opening windows
import matplotlib.pyplot as plt

os.makedirs('reports/figures', exist_ok=True)
df = pd.read_csv('data/creditcard.csv')

# 1) Class balance
counts = df['Class'].value_counts()
print('=== Class balance ===')
print(counts)
print('Fraud rate: {:.3%}'.format(df['Class'].mean()))

plt.figure(figsize=(5,4))
counts.plot(kind='bar', color=['#2E5496', '#C55A11'])
plt.title('Class balance (0 = legit, 1 = fraud)')
plt.ylabel('count'); plt.xticks(rotation=0); plt.tight_layout()
plt.savefig('reports/figures/class_balance.png', dpi=120); plt.close()

# 2) Amount: fraud vs legit
print()
print('=== Amount by class ===')
print(df.groupby('Class')['Amount'].describe()[['mean','50%','max']])

plt.figure(figsize=(6,4))
plt.hist(df[df.Class==0]['Amount'], bins=50, range=(0,500), alpha=0.6, label='legit', color='#2E5496')
plt.hist(df[df.Class==1]['Amount'], bins=50, range=(0,500), alpha=0.8, label='fraud', color='#C55A11')
plt.title('Transaction amount (0-500)'); plt.xlabel('Amount'); plt.ylabel('count')
plt.legend(); plt.tight_layout()
plt.savefig('reports/figures/amount_by_class.png', dpi=120); plt.close()

# 3) Which features separate fraud most (abs correlation with Class)
corr = df.corr(numeric_only=True)['Class'].drop('Class').abs().sort_values(ascending=False)
print()
print('=== Top 8 features most correlated with fraud ===')
print(corr.head(8))

plt.figure(figsize=(6,4))
corr.head(10).sort_values().plot(kind='barh', color='#2E5496')
plt.title('Top features correlated with fraud'); plt.xlabel('|correlation|'); plt.tight_layout()
plt.savefig('reports/figures/top_feature_corr.png', dpi=120); plt.close()

print()
print('Saved 3 charts to reports/figures/')
