import joblib
import numpy as np
import pandas as pd
import shap
from fastapi import FastAPI
from pydantic import create_model

# Load everything once at startup
model = joblib.load('models/xgb.joblib')
scaler = joblib.load('models/scaler.joblib')
FEATURES = joblib.load('models/feature_names.joblib')   # order: Time, V1..V28, Amount
THRESHOLD = joblib.load('models/threshold.joblib')
explainer = shap.TreeExplainer(model)

# Build a request schema with one float field per feature (auto-documented)
Transaction = create_model('Transaction', **{name: (float, ...) for name in FEATURES})

app = FastAPI(title='Fraud Detection API',
              description='Scores a card/e-wallet transaction and returns reason codes.',
              version='1.0.0')

@app.get('/')
def health():
    return {'status': 'ok', 'model': 'XGBoost+SMOTE', 'threshold': THRESHOLD}

@app.post('/score')
def score(txn: Transaction):
    data = txn.model_dump()
    row = pd.DataFrame([[data[f] for f in FEATURES]], columns=FEATURES)
    # scale raw Amount & Time exactly as in training
    row[['Amount', 'Time']] = scaler.transform(row[['Amount', 'Time']])

    prob = float(model.predict_proba(row)[0, 1])
    decision = 'FRAUD' if prob >= THRESHOLD else 'legit'

    sv = explainer.shap_values(row)[0]
    order = np.argsort(np.abs(sv))[::-1][:5]
    reasons = [{
        'feature': FEATURES[i],
        'value': round(float(row.iloc[0, i]), 3),
        'impact': round(float(sv[i]), 3),
        'pushes': 'fraud' if sv[i] > 0 else 'legit'
    } for i in order]

    return {
        'fraud_probability': round(prob, 4),
        'threshold': THRESHOLD,
        'decision': decision,
        'reason_codes': reasons
    }
