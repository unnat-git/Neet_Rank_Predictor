import streamlit as st
import numpy as np
import joblib
@st.cache_resource
def load_my_model():
    model = joblib.load("neet_predictor.joblib")
    return model

model = load_my_model()

st.title("🎓 NEET-UG Rank Predictor")
marks = st.number_input("Marks", min_value=300, max_value=720, value=650)
year = st.selectbox("Year", [2025, 2026])

if st.button("Predict"):
    if marks==720:
        st.success("Predicted Rank: 1")
    else:
        input_data = np.array([[marks, marks**2, year]])
        pred_log = model.predict(input_data)[0]
        final_rank = np.exp(pred_log) - 1
        st.success(f"Predicted Rank: {max(1, int(final_rank))}")
st.markdown(
    """
    <hr style="margin-top:50px;">
    <div style="text-align:center; font-size:14px; color:gray;">
        Built with ❤️ by Unnat  
        <br>
        <span style="font-size:12px;">
        ⚠️ Demo ML model — predictions are not official and may be inaccurate.
        </span>
    </div>
    """,
    unsafe_allow_html=True
)
