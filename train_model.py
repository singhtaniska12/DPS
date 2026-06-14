
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from imblearn.over_sampling import RandomOverSampler
import matplotlib.pyplot as plt
import seaborn as sns


TRAIN_CSV = "Training.csv"
TEST_CSV = "Testing.csv"
MODEL_OUT = "rf_disease_model.pkl"
ENCODER_OUT = "disease_label_encoder.pkl"
RANDOM_STATE = 42


if not os.path.exists(TRAIN_CSV):
    raise FileNotFoundError(f"{TRAIN_CSV} not found. Place it in the working directory.")

train_df = pd.read_csv(TRAIN_CSV)
print("Training data shape:", train_df.shape)


label_col = "prognosis" if "prognosis" in train_df.columns else train_df.columns[-1]
print("Label column detected as:", label_col)


X = train_df.drop(columns=[label_col])
y = train_df[label_col]


le = LabelEncoder()
y_enc = le.fit_transform(y)

plt.figure(figsize=(12, 5))
sns.countplot(x=y)
plt.xticks(rotation=90)
plt.title("Class distribution (Training)")
plt.tight_layout()
plt.show()


ros = RandomOverSampler(random_state=RANDOM_STATE)
X_res, y_res = ros.fit_resample(X, y_enc)
print("After resampling X:", X_res.shape, "y:", y_res.shape)


rf = RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
scores = cross_val_score(rf, X_res, y_res, cv=cv, scoring='accuracy', n_jobs=-1)
print("Cross-val accuracy scores:", scores)
print("Mean CV accuracy:", scores.mean())

rf.fit(X_res, y_res)
print("Model trained on resampled full training data.")


joblib.dump(rf, MODEL_OUT)
joblib.dump(le, ENCODER_OUT)
print(f"Saved model -> {MODEL_OUT}")
print(f"Saved label encoder -> {ENCODER_OUT}")


y_train_pred = rf.predict(X_res)
print("Train accuracy (resampled):", accuracy_score(y_res, y_train_pred))
print("\nClassification report (train):")
print(classification_report(y_res, y_train_pred, target_names=le.classes_.tolist()))


cm = confusion_matrix(y_res, y_train_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=False, fmt='d')
plt.title("Confusion matrix (train - resampled)")
plt.tight_layout()
plt.show()


if os.path.exists(TEST_CSV):
    test_df = pd.read_csv(TEST_CSV)
    print("Testing data shape:", test_df.shape)
    X_test = test_df.drop(columns=[label_col])
    y_test = le.transform(test_df[label_col]) 
    y_test_pred = rf.predict(X_test)
    print("Test accuracy:", accuracy_score(y_test, y_test_pred))
    print("\nClassification report (test):")
    print(classification_report(y_test, y_test_pred, target_names=le.classes_.tolist()))
else:
    print(f"{TEST_CSV} not found — skipping test evaluation.")


cols_out = "symptom_columns.pkl"
joblib.dump(X.columns.tolist(), cols_out)
print(f"Saved symptom columns -> {cols_out}")
