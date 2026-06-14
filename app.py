import streamlit as st
import joblib
import numpy as np
import pandas as pd


model = joblib.load("rf_disease_model.pkl")
encoder = joblib.load("disease_label_encoder.pkl")
symptom_columns = joblib.load("symptom_columns.pkl")


page_bg = """
<style>

@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600&display=swap');

* {
    font-family: 'Poppins', sans-serif;
}

/* Background with dark blur */
.stApp {
    background: rgba(10, 10, 10, 0.85);
    background-size: cover;
    backdrop-filter: blur(8px);
}

/* Glass container */
.main-card {
    background: rgba(20, 20, 20, 0.5);
    padding: 30px;
    border-radius: 18px;
    backdrop-filter: blur(12px);
    border: 1px solid rgba(255, 255, 255, 0.1);
    box-shadow: 0 4px 25px rgba(0,0,0,0.4);
    margin-top: 40px;
}

/* Heading styling */
.title {
    text-align: center;
    color: #00eaff;
    font-size: 38px;
    font-weight: 600;
    margin-bottom: 10px;
}

/* Subtitle */
.subtitle {
    text-align: center;
    color: #cccccc;
    font-size: 18px;
    margin-bottom: 30px;
}

/* Pretty button */
.stButton > button {
    background-color: #00eaff;
    color: black;
    border-radius: 10px;
    padding: 10px 20px;
    font-weight: 600;
    border: none;
    transition: 0.3s;
}

.stButton > button:hover {
    background-color: #00b7c2;
    transform: scale(1.03);
}

/* Prediction Text */
.pred-box {
    background: rgba(0, 255, 200, 0.15);
    padding: 20px;
    border-radius: 12px;
    border: 1px solid rgba(0,255,200,0.3);
    margin-top: 20px;
    color: #00ffd5;
    font-size: 22px;
    font-weight: 600;
    text-align: center;
}

</style>
"""

st.markdown(page_bg, unsafe_allow_html=True)


st.markdown("<div class='title'>Disease Prediction System</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Select symptoms and get an accurate disease prediction</div>", unsafe_allow_html=True)

with st.container():
    st.markdown("<div class='main-card'>", unsafe_allow_html=True)

    
    selected_symptoms = st.multiselect(
        "Choose Symptoms",
        symptom_columns,
        help="Start typing to search symptoms"
    )

          
    if st.button("Predict Disease"):

        if not selected_symptoms:
            st.warning("Please select at least one symptom.")
        else:
            input_vector = np.zeros(len(symptom_columns))

            for symptom in selected_symptoms:
                idx = symptom_columns.index(symptom)
                input_vector[idx] = 1

            input_df = pd.DataFrame([input_vector], columns=symptom_columns)

            prediction = encoder.inverse_transform(model.predict(input_df))[0]

            st.markdown(
                f"<div class='pred-box'>Predicted Disease: <br><br> {prediction}</div>",
                unsafe_allow_html=True
            )

    st.markdown("</div>", unsafe_allow_html=True)





