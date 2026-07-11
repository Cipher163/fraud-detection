import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, recall_score
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

X_train, X_test, y_train, y_test = joblib.load('data/processed/split.joblib')

def report(name, model):
    p = model.predict(X_test)
    print('--- {} ---'.format(name))
    print('Accuracy: {:.4f}'.format(accuracy_score(y_test, p)))
    print('Recall (fraud caught): {:.4f}'.format(recall_score(y_test, p)))
    print()

# 0) The trap: always predict legit
dummy = DummyClassifier(strategy='most_frequent').fit(X_train, y_train)
report('Dummy (always legit)', dummy)

# 1) Baseline: logistic regression with balanced class weights
logreg = LogisticRegression(max_iter=1000, class_weight='balanced')
logreg.fit(X_train, y_train)
joblib.dump(logreg, 'models/logreg.joblib')
report('Logistic Regression (balanced)', logreg)

# 2) XGBoost + SMOTE (your headline model)
sm = SMOTE(random_state=42)
X_res, y_res = sm.fit_resample(X_train, y_train)
X_res = pd.DataFrame(X_res, columns=X_train.columns)
print('After SMOTE, train class counts:', np.bincount(y_res.astype(int)))
print()

xgb = XGBClassifier(
    n_estimators=300, max_depth=6, learning_rate=0.1,
    subsample=0.9, colsample_bytree=0.9,
    eval_metric='aucpr', n_jobs=-1, random_state=42
)
xgb.fit(X_res, y_res)
joblib.dump(xgb, 'models/xgb.joblib')
joblib.dump(list(X_train.columns), 'models/feature_names.joblib')
report('XGBoost + SMOTE', xgb)

print('Saved models -> models/logreg.joblib, models/xgb.joblib')
