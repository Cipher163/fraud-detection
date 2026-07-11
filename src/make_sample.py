import json, joblib
import pandas as pd

raw = pd.read_csv('data/creditcard.csv')
_, X_test, _, y_test = joblib.load('data/processed/split.joblib')

fraud_idx = y_test[y_test == 1].index[0]
legit_idx = y_test[y_test == 0].index[0]

for name, idx in [('fraud', fraud_idx), ('legit', legit_idx)]:
    row = raw.loc[idx].drop(labels=['Class'])
    payload = {k: float(v) for k, v in row.items()}
    with open('sample_{}.json'.format(name), 'w') as f:
        json.dump(payload, f, indent=2)
    print('Wrote sample_{}.json (true label = {})'.format(name, name))
