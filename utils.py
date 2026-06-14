import joblib
import numpy as np

def load_artifacts():
    model = joblib.load("rf_disease_model.pkl")
    encoder = joblib.load("disease_label_encoder.pkl")
    symptom_cols = joblib.load("symptom_columns.pkl")
    return model, encoder, symptom_cols


def predict_disease(selected_symptoms, symptom_cols, model, encoder):
    # Input vector (same length as training features)
    input_data = np.zeros(len(symptom_cols))

    # Set 1 for selected symptoms
    for symptom in selected_symptoms:
        if symptom in symptom_cols:
            index = symptom_cols.index(symptom)
            input_data[index] = 1

    # Reshape for prediction
    input_data = input_data.reshape(1, -1)

    # Predict
    prediction_encoded = model.predict(input_data)[0]

    # Decode to actual disease name
    prediction = encoder.inverse_transform([prediction_encoded])[0]

    return prediction

