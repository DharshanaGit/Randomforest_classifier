import streamlit as st
import pandas as pd
import joblib

# -------------------------------
# Page configuration
# -------------------------------
st.set_page_config(
    page_title="Titanic Survival Predictor",
    page_icon="🚢",
    layout="centered"
)

# -------------------------------
# Load trained model
# -------------------------------
@st.cache_resource
def load_model():
    return joblib.load("titanic_rf_model.joblib")

model = load_model()

# -------------------------------
# App Title
# -------------------------------
st.title("🚢 Titanic Survival Prediction App")
st.write("Enter passenger details to predict survival outcome.")

# -------------------------------
# User Inputs
# -------------------------------
Pclass = st.selectbox("Passenger Class", [1, 2, 3])
Sex = st.selectbox("Sex", ["Male", "Female"])
Age = st.number_input("Age", min_value=0, max_value=100, value=25)
SibSp = st.number_input("Siblings / Spouses Aboard", min_value=0, max_value=10, value=0)
Parch = st.number_input("Parents / Children Aboard", min_value=0, max_value=10, value=0)
Fare = st.number_input("Fare", min_value=0.0, value=32.0)

# -------------------------------
# Encoding (MUST MATCH TRAINING)
# -------------------------------
Sex_encoded = 1 if Sex == "Male" else 0

# -------------------------------
# Create input DataFrame
# -------------------------------
input_data = pd.DataFrame(
    [[Pclass, Sex_encoded, Age, SibSp, Parch, Fare]],
    columns=['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']
)

# -------------------------------
# Prediction
# -------------------------------
if st.button("Predict Survival"):
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    if prediction == 1:
        st.success(f"🎉 Passenger Survived (Confidence: {probability:.2%})")
    else:
        st.error(f"❌ Passenger Did Not Survive (Confidence: {1 - probability:.2%})")

# -------------------------------
# Show input data (optional)
# -------------------------------