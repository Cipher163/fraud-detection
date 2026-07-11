import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
import joblib

os.makedirs('data/processed', exist_ok=True)
os.makedirs('models', exist_ok=True)

df = pd.read_csv('data/creditcard.csv')

X = df.drop(columns=['Class'])
y = df['Class']

# Stratified split: keep the fraud ratio identical in train and test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=42
)

# Scale ONLY Amount and Time, fit on TRAIN only (no leakage)
scaler = RobustScaler()
cols_to_scale = ['Amount', 'Time']
X_train = X_train.copy()
X_test = X_test.copy()
X_train[cols_to_scale] = scaler.fit_transform(X_train[cols_to_scale])
X_test[cols_to_scale]  = scaler.transform(X_test[cols_to_scale])

# Save the split and the fitted scaler for later chunks
joblib.dump((X_train, X_test, y_train, y_test), 'data/processed/split.joblib')
joblib.dump(scaler, 'models/scaler.joblib')

print('Train shape:', X_train.shape, '| frauds in train:', int(y_train.sum()))
print('Test  shape:', X_test.shape,  '| frauds in test :', int(y_test.sum()))
print('Train fraud rate: {:.3%}'.format(y_train.mean()))
print('Test  fraud rate: {:.3%}'.format(y_test.mean()))
print('Saved split -> data/processed/split.joblib')
print('Saved scaler -> models/scaler.joblib')
