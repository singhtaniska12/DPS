import streamlit as st
import joblib
import numpy as np
import pandas as pd

# ==========================
# PAGE CONFIG
# ==========================
st.set_page_config(
    page_title=" Disease Prediction System",
    page_icon="🏥",
    layout="wide"
)

# ==========================
# LOAD MODEL
# ==========================

model = joblib.load("rf_disease_model.pkl")
encoder = joblib.load("disease_label_encoder.pkl")
symptom_columns = joblib.load("symptom_columns.pkl")
with st.sidebar:

    st.title("🏥 Health Dashboard")

    st.metric("Diseases", "41")

    st.metric("Symptoms", len(symptom_columns))

    st.metric("Model", "Random Forest")

    st.markdown("---")


    st.success("🟢 System Online")

    st.caption(
    "Machine Learning powered disease prediction dashboard."
)

    st.info(
        "Select symptoms and predict diseases using AI."
    )
disease_info = {
    "Diabetes ": {
        "description": "A condition that affects blood sugar regulation.",
        "precautions": [
            "Exercise regularly",
            "Maintain healthy diet",
            "Monitor blood sugar",
            "Visit doctor regularly"
        ]
    },

    "Dengue": {
        "description": "Mosquito-borne viral infection causing fever and body pain.",
        "precautions": [
            "Stay hydrated",
            "Take proper rest",
            "Avoid mosquito exposure",
            "Consult a doctor"
        ]
    },

    "Malaria": {
        "description": "Parasitic infection spread by mosquitoes.",
        "precautions": [
            "Use mosquito nets",
            "Take prescribed medication",
            "Stay hydrated",
            "Seek medical care"
        ]
    },
    "Typhoid": {
        "description": "A bacterial infection caused by contaminated food and water.",
        "precautions": [
            "Drink clean water",
            "Maintain hygiene",
            "Take prescribed antibiotics",
            "Get adequate rest"
        ]
    },

"Jaundice": {
    "description": "A condition causing yellowing of the skin and eyes due to liver dysfunction.",
    "precautions": [
        "Avoid alcohol",
        "Stay hydrated",
        "Eat healthy food",
        "Consult a doctor"
    ]
},

"Migraine": {
    "description": "A neurological disorder causing severe headaches and sensitivity to light.",
    "precautions": [
        "Avoid stress",
        "Maintain sleep schedule",
        "Stay hydrated",
        "Avoid trigger foods"
    ]
},

"Pneumonia": {
    "description": "An infection that inflames the air sacs in one or both lungs.",
    "precautions": [
        "Take medications as prescribed",
        "Get adequate rest",
        "Stay hydrated",
        "Avoid smoking"
    ]
},

"Common Cold": {
    "description": "A viral infection affecting the upper respiratory tract.",
    "precautions": [
        "Rest properly",
        "Drink warm fluids",
        "Maintain hygiene",
        "Avoid close contact with infected individuals"
    ]
},

"Chicken pox": {
    "description": "A contagious viral infection causing itchy blisters and rash.",
    "precautions": [
        "Avoid scratching",
        "Get proper rest",
        "Maintain hygiene",
        "Consult a healthcare provider"
    ]
},

"Tuberculosis": {
    "description": "A bacterial infection that mainly affects the lungs.",
    "precautions": [
        "Complete medication course",
        "Wear a mask",
        "Maintain nutrition",
        "Follow medical advice"
    ]
},

"Heart attack": {
    "description": "A medical emergency caused by blocked blood flow to the heart.",
    "precautions": [
        "Maintain healthy diet",
        "Exercise regularly",
        "Avoid smoking",
        "Monitor blood pressure"
    ]
},

"Hypertension ": {
    "description": "A condition in which blood pressure remains consistently high.",
    "precautions": [
        "Reduce salt intake",
        "Exercise regularly",
        "Manage stress",
        "Monitor blood pressure"
    ]
},

"Hypothyroidism": {
    "description": "A condition where the thyroid gland does not produce enough hormones.",
    "precautions": [
        "Take prescribed medication",
        "Monitor thyroid levels",
        "Maintain balanced diet",
        "Exercise regularly"
    ]
},

"Hyperthyroidism": {
    "description": "A condition where the thyroid gland produces excessive hormones.",
    "precautions": [
        "Take prescribed medicines",
        "Avoid excessive iodine intake",
        "Regular medical checkups",
        "Manage stress"
    ]
},

"Bronchial Asthma": {
    "description": "A respiratory condition causing breathing difficulties.",
    "precautions": [
        "Avoid allergens",
        "Carry inhaler",
        "Avoid smoking",
        "Follow treatment plan"
    ]
},

"Urinary tract infection": {
    "description": "An infection affecting any part of the urinary system.",
    "precautions": [
        "Drink plenty of water",
        "Maintain hygiene",
        "Take prescribed antibiotics",
        "Avoid holding urine"
    ]
},

"Gastroenteritis": {
    "description": "Inflammation of the stomach and intestines causing diarrhea and vomiting.",
    "precautions": [
        "Stay hydrated",
        "Eat light meals",
        "Maintain hygiene",
        "Get adequate rest"
    ]
},

"Arthritis": {
    "description": "Inflammation of one or more joints causing pain and stiffness.",
    "precautions": [
        "Exercise regularly",
        "Maintain healthy weight",
        "Take prescribed medicines",
        "Use physical therapy if advised"
    ]
},

"Acne": {
    "description": "A skin condition that occurs when hair follicles become clogged with oil and dead skin cells.",
    "precautions": [
        "Wash face regularly",
        "Avoid oily products",
        "Stay hydrated",
        "Follow dermatologist advice"
    ]
},

"Allergy": {
    "description": "An immune system reaction to a foreign substance.",
    "precautions": [
        "Avoid allergens",
        "Take antihistamines",
        "Keep surroundings clean",
        "Consult a doctor if symptoms worsen"
    ]
},

"AIDS": {
    "description": "A chronic condition caused by HIV that weakens the immune system.",
    "precautions": [
        "Take antiretroviral therapy",
        "Practice safe hygiene",
        "Maintain nutrition",
        "Regular medical checkups"
    ]
},

"Alcoholic hepatitis": {
    "description": "Liver inflammation caused by excessive alcohol consumption.",
    "precautions": [
        "Avoid alcohol completely",
        "Eat nutritious food",
        "Stay hydrated",
        "Follow medical advice"
    ]
},

"Cervical spondylosis": {
    "description": "Age-related wear and tear affecting spinal disks in the neck.",
    "precautions": [
        "Maintain proper posture",
        "Perform neck exercises",
        "Avoid heavy lifting",
        "Consult a physiotherapist"
    ]
},
 "Chronic cholestasis": {
    "description": "A liver condition that reduces or blocks the flow of bile.",
    "precautions": [
        "Follow a healthy diet",
        "Avoid alcohol",
        "Take prescribed medications",
        "Regular liver checkups"
    ]
},

"Dimorphic hemmorhoids(piles)": {
    "description": "Swollen veins in the lower rectum and anus.",
    "precautions": [
        "Eat high-fiber foods",
        "Drink plenty of water",
        "Avoid prolonged sitting",
        "Exercise regularly"
    ]
},

"Drug Reaction": {
    "description": "An unwanted response caused by certain medications.",
    "precautions": [
        "Stop medication if advised",
        "Consult a doctor",
        "Avoid self-medication",
        "Monitor symptoms carefully"
    ]
},

"Fungal infection": {
    "description": "An infection caused by fungi affecting skin, nails, or other tissues.",
    "precautions": [
        "Keep skin dry",
        "Maintain hygiene",
        "Use antifungal medication",
        "Avoid sharing personal items"
    ]
},

"GERD": {
    "description": "A digestive disorder where stomach acid frequently flows back into the esophagus.",
    "precautions": [
        "Avoid spicy foods",
        "Eat smaller meals",
        "Maintain healthy weight",
        "Avoid lying down after meals"
    ]
},

"Hepatitis B": {
    "description": "A viral infection that affects the liver.",
    "precautions": [
        "Avoid alcohol",
        "Get regular checkups",
        "Follow prescribed treatment",
        "Practice safe hygiene"
    ]
},

"Hepatitis C": {
    "description": "A viral infection that causes liver inflammation.",
    "precautions": [
        "Avoid alcohol",
        "Take antiviral medication",
        "Maintain healthy diet",
        "Regular medical monitoring"
    ]
},

"Hepatitis D": {
    "description": "A serious liver infection that occurs only with Hepatitis B.",
    "precautions": [
        "Monitor liver health",
        "Follow medical advice",
        "Avoid alcohol",
        "Maintain healthy lifestyle"
    ]
},

"Hepatitis E": {
    "description": "A viral liver disease often spread through contaminated water.",
    "precautions": [
        "Drink clean water",
        "Maintain hygiene",
        "Eat safe food",
        "Get proper rest"
    ]
},

"hepatitis A": {
    "description": "A viral infection affecting liver function.",
    "precautions": [
        "Wash hands regularly",
        "Drink clean water",
        "Avoid contaminated food",
        "Get adequate rest"
    ]
},

"Hypoglycemia": {
    "description": "A condition characterized by abnormally low blood sugar levels.",
    "precautions": [
        "Monitor blood sugar",
        "Eat regular meals",
        "Carry glucose source",
        "Follow medical advice"
    ]
},

"Impetigo": {
    "description": "A highly contagious bacterial skin infection.",
    "precautions": [
        "Maintain hygiene",
        "Avoid scratching",
        "Use prescribed medication",
        "Avoid sharing personal items"
    ]
},

"Osteoarthristis": {
    "description": "A degenerative joint disease causing cartilage breakdown.",
    "precautions": [
        "Exercise regularly",
        "Maintain healthy weight",
        "Protect joints",
        "Follow physiotherapy guidance"
    ]
},

"Paralysis (brain hemorrhage)": {
    "description": "Loss of muscle function due to bleeding in the brain.",
    "precautions": [
        "Seek immediate medical care",
        "Attend rehabilitation",
        "Monitor blood pressure",
        "Follow treatment plan"
    ]
},

"Peptic ulcer diseae": {
    "description": "Open sores that develop on the lining of the stomach or intestine.",
    "precautions": [
        "Avoid spicy food",
        "Limit alcohol",
        "Take prescribed medication",
        "Reduce stress"
    ]
},

"Psoriasis": {
    "description": "A chronic autoimmune condition affecting the skin.",
    "precautions": [
        "Keep skin moisturized",
        "Avoid skin injuries",
        "Manage stress",
        "Follow dermatologist advice"
    ]
},

"Varicose veins": {
    "description": "Enlarged and twisted veins commonly found in the legs.",
    "precautions": [
        "Exercise regularly",
        "Avoid prolonged standing",
        "Elevate legs when resting",
        "Maintain healthy weight"
    ]
},

"Diabetes": {
    "description": "A condition that affects blood sugar regulation.",
    "precautions": [
        "Exercise regularly",
        "Maintain healthy diet",
        "Monitor blood sugar",
        "Visit doctor regularly"
    ]
},

"Diabetes ": {
    "description": "A condition that affects blood sugar regulation.",
    "precautions": [
        "Exercise regularly",
        "Maintain healthy diet",
        "Monitor blood sugar",
        "Visit doctor regularly"
    ]
},

"Dengue": {
    "description": "A mosquito-borne viral infection causing high fever and severe body pain.",
    "precautions": [
        "Stay hydrated",
        "Take proper rest",
        "Avoid mosquito exposure",
        "Consult a doctor"
    ]
},

"Malaria": {
    "description": "A parasitic infection transmitted through mosquito bites.",
    "precautions": [
        "Use mosquito nets",
        "Take prescribed medication",
        "Stay hydrated",
        "Seek medical care"
    ]
}

}

# ==========================
# CUSTOM CSS
# ==========================
st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="css"]{
    font-family:'Poppins',sans-serif;
}

.stApp{
    background: linear-gradient(
        -45deg,
        #0f172a,
        #1e3a8a,
        #2563eb,
        #06b6d4
    );
    background-size:400% 400%;
    animation:gradientBG 15s ease infinite;
}

@keyframes gradientBG{
    0%{background-position:0% 50%;}
    50%{background-position:100% 50%;}
    100%{background-position:0% 50%;}
}

/* Hero Title */

.hero-title{
    text-align:center;
    font-size:62px;
    font-weight:800;
    color:white;
    margin-top:10px;
    text-shadow:0 0 30px rgba(255,255,255,0.5);
}

.hero-sub{
    text-align:center;
    color:#e2e8f0;
    font-size:20px;
    margin-bottom:40px;
}

/* Glass Card */

.glass-card{
    background: rgba(255,255,255,0.12);
    backdrop-filter: blur(18px);
    -webkit-backdrop-filter: blur(18px);

    border:1px solid rgba(255,255,255,0.2);

    border-radius:24px;
    padding:25px;

    color:white;

    transition:0.4s;
}

.glass-card:hover{
    transform:translateY(-8px);
    box-shadow:0 15px 40px rgba(0,0,0,0.3);
}

/* Prediction Box */

.pred-box{
    background: rgba(255,255,255,0.15);

    backdrop-filter: blur(20px);

    border-radius:25px;

    padding:35px;

    text-align:center;

    border:1px solid rgba(255,255,255,0.25);

    color:white;

    animation:fadeIn 1s ease;
}

@keyframes fadeIn{
    from{
        opacity:0;
        transform:translateY(20px);
    }
    to{
        opacity:1;
        transform:translateY(0px);
    }
}

/* Buttons */

.stButton > button{
    width:100%;
    height:60px;

    border:none;

    border-radius:15px;

    font-size:20px;

    font-weight:700;

    color:white;

    background:linear-gradient(
    135deg,
    #00c6ff,
    #0072ff
    );
}

/* Metric Cards */

.metric-box{
    background:rgba(255,255,255,0.12);

    border-radius:20px;

    padding:25px;

    text-align:center;

    color:white;

    border:1px solid rgba(255,255,255,0.2);
}

.metric-number{
    font-size:40px;
    font-weight:800;
}

.metric-title{
    font-size:16px;
    color:#e2e8f0;
}

</style>
""", unsafe_allow_html=True)


# ==========================
# HEADER
# ==========================
st.markdown("""
<div class="hero-title">
Disease Prediction System
</div>

<div class="hero-sub">
AI-powered symptom analysis for preliminary disease prediction
</div>
""", unsafe_allow_html=True)

# ==========================
# FEATURE CARDS
# ==========================
col1,col2,col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class='glass-card'>
    <h1>⚡</h1>
    <h3>Fast Prediction</h3>
    <p>Instant Disease Detection</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class='glass-card'>
    <h1>🤖</h1>
    <h3>AI Powered</h3>
    <p>Machine Learning Based Analysis</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class='glass-card'>
    <h1>🔒</h1>
    <h3>Secure</h3>
    <p>Privacy Focused Prediction</p>
    </div>
    """, unsafe_allow_html=True)
# ==========================
# SYMPTOM INPUT
# ==========================
selected_symptoms = st.multiselect(
    "🔍 Select Symptoms",
    symptom_columns,
    help="Search and select one or more symptoms"
)

# ==========================
# STATS
# ==========================
c1,c2,c3 = st.columns(3)

with c1:
    st.markdown("""
    <div class='metric-box'>
        <div class='metric-number'>41</div>
        <div class='metric-title'>Diseases</div>
    </div>
    """, unsafe_allow_html=True)

with c2:
    st.markdown(f"""
    <div class='metric-box'>
        <div class='metric-number'>{len(symptom_columns)}</div>
        <div class='metric-title'>Symptoms</div>
    </div>
    """, unsafe_allow_html=True)

with c3:
    st.markdown("""
    <div class='metric-box'>
        <div class='metric-number'>AI</div>
        <div class='metric-title'>Random Forest</div>
    </div>
    """, unsafe_allow_html=True)
# ==========================
# PREDICTION
# ==========================
if st.button("🩺 Predict Disease"):
    with st.spinner("🔍 Analyzing symptoms..."):
     import time
     time.sleep(2)

    if len(selected_symptoms) == 0:
        st.warning("Please select at least one symptom.")

    else:

        input_vector = np.zeros(len(symptom_columns))

        for symptom in selected_symptoms:
            idx = symptom_columns.index(symptom)
            input_vector[idx] = 1

        input_df = pd.DataFrame(
            [input_vector],
            columns=symptom_columns
        )

        prediction = encoder.inverse_transform(
            model.predict(input_df)
        )[0]
        probs = model.predict_proba(input_df)[0]

        top3_idx = np.argsort(probs)[-3:][::-1]

        st.subheader("🏆 Top 3 Possible Diseases")

        for idx in top3_idx:
            disease = encoder.inverse_transform([idx])[0]
            score = probs[idx] * 100

            st.write(f"🔹 {disease} — {score:.2f}%")

        confidence = np.max(probs) * 100

        st.markdown("### 🎯 Prediction Confidence")
        st.progress(int(confidence))

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Confidence", f"{confidence:.2f}%")
        with col2:
            st.metric("Selected Symptoms", len(selected_symptoms))

        st.success("Prediction Completed Successfully")
        st.toast("Prediction Completed Successfully")

        st.markdown(
            f"""
            <div class='pred-box'>
                🩺 Predicted Disease
                <br><br>
                {prediction}
            </div>
            """,
            unsafe_allow_html=True
        )

        st.write("")

        if prediction in disease_info:
            st.subheader("📖 Disease Information")
            st.write(disease_info[prediction]["description"])

            st.subheader("🛡️ Precautions")
            for precaution in disease_info[prediction]["precautions"]:
                st.write("✅", precaution)
        else:
            st.subheader("📖 Disease Information")
            st.info(
                f"""
                Disease Detected: {prediction}

                Detailed disease information is currently unavailable.

                Please consult a healthcare professional
                for proper diagnosis and treatment.
                """
            )

        st.info(
            "⚠️ This prediction is generated using a Machine Learning model and should not replace professional medical advice."
        )
        report = f"""
 Disease Prediction Report

Predicted Disease:
{prediction}

Confidence:
{confidence:.2f}%
if confidence > 90:
    st.success("🟢 High Confidence Prediction")
elif confidence > 70:
    st.warning("🟡 Medium Confidence Prediction")
else:
    st.error("🔴 Low Confidence Prediction")

Symptoms:
{', '.join(selected_symptoms)}
"""


# ==========================
# FOOTER
# ==========================

st.write("")
st.write("")

st.markdown("---")

st.markdown(
    """
    <center>
    <h4 style='color:#94a3b8;'>
    Developed using Python, Streamlit & Machine Learning
    </h4>
    </center>
    """,
    unsafe_allow_html=True
)
