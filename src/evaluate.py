import joblib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import (average_precision_score, roc_auc_score,
    classification_report, confusion_matrix, precision_recall_curve,
    precision_score, recall_score, f1_score)

X_train, X_test, y_train, y_test = joblib.load('data/processed/split.joblib')
logreg = joblib.load('models/logreg.joblib')
xgb = joblib.load('models/xgb.joblib')

# Probability of fraud (class 1) from each model
proba_lr = logreg.predict_proba(X_test)[:, 1]
proba_xgb = xgb.predict_proba(X_test)[:, 1]

print('=== Headline metrics (higher = better) ===')
for name, proba in [('LogReg', proba_lr), ('XGBoost', proba_xgb)]:
    print('{:8s}  PR-AUC={:.4f}   ROC-AUC={:.4f}'.format(
        name, average_precision_score(y_test, proba), roc_auc_score(y_test, proba)))

print()
print('=== XGBoost @ default threshold 0.5 ===')
pred05 = (proba_xgb >= 0.5).astype(int)
print(classification_report(y_test, pred05, digits=4, target_names=['legit','fraud']))
print('Confusion matrix [rows=actual, cols=predicted]:')
print(confusion_matrix(y_test, pred05))

# Threshold sweep for XGBoost
print()
print('=== XGBoost threshold sweep ===')
print('thresh   precision   recall     F1')
best_t, best_f1 = 0.5, -1
for t in np.arange(0.10, 0.95, 0.05):
    pred = (proba_xgb >= t).astype(int)
    p = precision_score(y_test, pred, zero_division=0)
    r = recall_score(y_test, pred)
    f = f1_score(y_test, pred, zero_division=0)
    print('{:.2f}      {:.4f}     {:.4f}   {:.4f}'.format(t, p, r, f))
    if f > best_f1:
        best_f1, best_t = f, t

print()
print('Best-F1 threshold = {:.2f} (F1={:.4f})'.format(best_t, best_f1))
joblib.dump(float(best_t), 'models/threshold.joblib')

# Precision-Recall curve
prec, rec, _ = precision_recall_curve(y_test, proba_xgb)
plt.figure(figsize=(6,4))
plt.plot(rec, prec, color='#2E5496')
plt.xlabel('Recall'); plt.ylabel('Precision')
plt.title('XGBoost Precision-Recall curve (PR-AUC={:.3f})'.format(
    average_precision_score(y_test, proba_xgb)))
plt.grid(alpha=0.3); plt.tight_layout()
plt.savefig('reports/figures/pr_curve.png', dpi=120); plt.close()
print('Saved PR curve -> reports/figures/pr_curve.png')
print('Saved chosen threshold -> models/threshold.joblib')
