import streamlit as st
import numpy as np
import joblib
@st.cache_resource
def load_my_model():
    model = joblib.load("neet_predictor.joblib")
    return model

model = load_my_model()

st.title("🎓 NEET Rank Predictor")
marks = st.number_input("Marks", 0, 720, 650)
year = st.selectbox("Year", [2025, 2026])

if st.button("Predict"):
    input_data = np.array([[marks, marks**2, year]])
    pred_log = model.predict(input_data)[0]
    final_rank = np.exp(pred_log) - 1
    st.success(f"Predicted Rank: {max(1, int(final_rank))}")