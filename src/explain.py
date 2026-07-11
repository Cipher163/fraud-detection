import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import shap

X_train, X_test, y_train, y_test = joblib.load('data/processed/split.joblib')
xgb = joblib.load('models/xgb.joblib')

# TreeExplainer is fast and exact for tree models like XGBoost
explainer = shap.TreeExplainer(xgb)

# --- Global view: importance across a sample of the test set ---
sample = X_test.sample(2000, random_state=42)
shap_values = explainer.shap_values(sample)

plt.figure()
shap.summary_plot(shap_values, sample, plot_type='bar', show=False, max_display=12)
plt.tight_layout()
plt.savefig('reports/figures/shap_global.png', dpi=120, bbox_inches='tight'); plt.close()
print('Saved global importance -> reports/figures/shap_global.png')

# --- Local view: explain a few ACTUAL fraud transactions ---
def reason_codes(row_df, top_k=5):
    sv = explainer.shap_values(row_df)[0]          # contributions for this row
    feats = row_df.columns
    order = np.argsort(np.abs(sv))[::-1][:top_k]   # biggest absolute impact first
    out = []
    for i in order:
        out.append({
            'feature': feats[i],
            'value': round(float(row_df.iloc[0, i]), 3),
            'shap': round(float(sv[i]), 3),
            'pushes': 'FRAUD' if sv[i] > 0 else 'legit'
        })
    return out

fraud_idx = y_test[y_test == 1].index[:3]
for idx in fraud_idx:
    row = X_test.loc[[idx]]
    prob = float(xgb.predict_proba(row)[0, 1])
    print()
    print('Transaction {} | fraud probability = {:.3f}'.format(idx, prob))
    for rc in reason_codes(row):
        print('   {:6s}  value={:>8}  shap={:>7}  -> {}'.format(
            rc['feature'], rc['value'], rc['shap'], rc['pushes']))
