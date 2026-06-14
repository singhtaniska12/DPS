import pandas as pd
import joblib

df = pd.read_csv("Training.csv")
symptom_cols = list(df.drop("prognosis", axis=1).columns)

joblib.dump(symptom_cols, "symptom_columns.pkl")

print("symptom_columns.pkl created successfully!")
