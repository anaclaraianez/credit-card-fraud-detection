import pandas as pd

url = "https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv"

df = pd.read_csv(url)

print(df.head())
print(df["Class"].value_counts(normalize=True))

import numpy as np
df["Amount_log"] = np.log1p(df["Amount"])

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
df["Amount_scaled"] = scaler.fit_transform(df[["Amount"]])

from sklearn.model_selection import train_test_split
X = df.drop("Class", axis=1)
y = df["Class"]
X_train, X_test, y_train, y_test = train_test_split(
     X, y, stratify=y, test_size=0.3, random_state=42 )

from sklearn.linear_model import LogisticRegression
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

from sklearn.metrics import classification_report 
print (classification_report(y_test, y_pred))

from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt

y_probs = model.predict_proba(X_test)[:, 1]

fpr, tpr, _ = roc_curve(y_test, y_probs)

plt.plot(fpr, tpr)
plt.title("ROC Curve")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")

plt.show()

print("AUC:", roc_auc_score(y_test, y_probs))

from sklearn.metrics import precision_recall_curve

precision, recall, _ = precision_recall_curve(y_test, y_probs)

plt.plot(recall, precision)
plt.title("Precision-Recall Curve")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.show()

from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

# UNDERSAMPLING
under = RandomUnderSampler(random_state=42)

X_train_under, y_train_under = under.fit_resample(
    X_train,
    y_train
)

model_under = LogisticRegression(max_iter=1000)

model_under.fit(
    X_train_under,
    y_train_under
)

y_pred_under = model_under.predict(X_test)

print("\n--- RESULTADO UNDERSAMPLING ---")
print(classification_report(y_test, y_pred_under))

# OVERSAMPLING COM SMOTE
smote = SMOTE(random_state=42)

X_train_smote, y_train_smote = smote.fit_resample(
    X_train,
    y_train
)

model_smote = LogisticRegression(max_iter=1000)

model_smote.fit(
    X_train_smote,
    y_train_smote
)

y_pred_smote = model_smote.predict(X_test)

print("\n--- RESULTADO SMOTE ---")
print(classification_report(y_test, y_pred_smote))

from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(
    n_estimators=50,
    max_depth=10,
    class_weight="balanced",
    n_jobs=-1,
    random_state=42
)

rf.fit(X_train, y_train)

y_pred_rf = rf.predict(X_test)

print(classification_report(y_test, y_pred_rf))

from sklearn.pipeline import Pipeline

pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(max_iter=1000))
])

pipeline.fit(X_train, y_train)

y_pred = pipeline.predict(X_test)

threshold = 0.3

y_pred_custom = (y_probs > threshold).astype(int)

print(classification_report(y_test, y_pred_custom))

from xgboost import XGBClassifier

xgb = XGBClassifier(
    scale_pos_weight=10,
    use_label_encoder=False,
    eval_metric="logloss"
)

xgb.fit(X_train, y_train)

y_pred_xgb = xgb.predict(X_test)

print("\n--- RESULTADO XGBOOST ---")
print(classification_report(y_test, y_pred_xgb))

import matplotlib.pyplot as plt
import pandas as pd

importancias = pd.DataFrame({
    "Variavel": X.columns,
    "Importancia": xgb.feature_importances_
})

importancias = importancias.sort_values(
    by="Importancia",
    ascending=False
).head(15)

plt.figure(figsize=(10, 6))

plt.barh(
    importancias["Variavel"],
    importancias["Importancia"],
    color="teal"
)

plt.gca().invert_yaxis()

plt.title("15 Variáveis Mais Importantes - XGBoost")
plt.xlabel("Importância")
plt.ylabel("Variável")

plt.tight_layout()
plt.show()

from sklearn.model_selection import GridSearchCV

param_grid = {
    "max_depth": [3, 5],
    "n_estimators": [50, 100]
}

grid = GridSearchCV(
    XGBClassifier(eval_metric="logloss"),
    param_grid,
    scoring="recall",
    cv=3
)

grid.fit(X_train, y_train)

print("Melhor modelo:", grid.best_params_)

import shap

melhor_modelo = grid.best_estimator_

X_shap = X_test.sample(
    n=min(1000, len(X_test)),
    random_state=42
)

explainer = shap.TreeExplainer(melhor_modelo)

shap_values = explainer.shap_values(X_shap)

shap.summary_plot(
    shap_values,
    X_shap,
    max_display=15
)

grid.fit(X_train, y_train)

print("Melhor modelo:", grid.best_params_)