import pandas as pd

df = pd.read_csv('data/creditcard.csv')

print('Shape:', df.shape)
print('Columns:', list(df.columns))
print('Class balance:')
print(df['Class'].value_counts())
print('Fraud rate: {:.3%}'.format(df['Class'].mean()))
print('Missing values:', df.isnull().sum().sum())
