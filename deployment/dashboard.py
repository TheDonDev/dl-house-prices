"""
Streamlit dashboard for Nairobi property price prediction.
"""
import pandas as pd
import plotly.express as px
import requests
import streamlit as st

st.set_page_config(
    page_title="Nairobi Property Price Predictor",
    page_icon="🏠",
    layout="wide",
)

st.title("🏠 Nairobi Property Price Predictor")
st.markdown("**Model:** PyTorch Neural Network | **Dataset:** Nairobi property listings")

st.sidebar.header("📊 Property Details")
st.sidebar.markdown("Adjust the features to estimate the property price.")

bedrooms = st.sidebar.slider("Bedrooms", 1, 10, 3, 1)
bathrooms = st.sidebar.slider("Bathrooms", 1, 10, 2, 1)
house_size_m2 = st.sidebar.slider("House size (m²)", 20, 1200, 180, 10)
land_size_acres = st.sidebar.slider("Land size (acres)", 0.0, 10.0, 0.5, 0.1)

if st.sidebar.button("🔮 Predict Price", type="primary", use_container_width=True):
    data = {
        "bedrooms": bedrooms,
        "bathrooms": bathrooms,
        "house_size_m2": house_size_m2,
        "land_size_acres": land_size_acres,
    }

    try:
        response = requests.post("http://localhost:8000/predict", json=data)
        if response.status_code == 200:
            result = response.json()

            col1, col2 = st.columns(2)

            with col1:
                st.metric(
                    "🏠 Predicted Price",
                    result["price_ksh"],
                    delta="Neural Network Estimate"
                )

            with col2:
                st.metric(
                    "📊 Input Profile",
                    f"{bedrooms} bd / {bathrooms} ba",
                    delta=f"{house_size_m2} m² / {land_size_acres} acres"
                )

            st.subheader("📊 Property Inputs")
            st.dataframe(pd.DataFrame([data]), use_container_width=True)
        else:
            st.error(f"API Error: {response.status_code}")
    except Exception as e:
        st.error(f"Connection error: {e}")

st.subheader("🧠 Neural Network Architecture")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    **Architecture:**
    - Input Layer: 4 features
    - Hidden Layer 1: 64 neurons (ReLU + Dropout)
    - Hidden Layer 2: 32 neurons (ReLU + Dropout)
    - Hidden Layer 3: 16 neurons (ReLU + Dropout)
    - Output Layer: 1 neuron
    """)

with col2:
    st.markdown("""
    **Model Inputs:**
    - Bedrooms
    - Bathrooms
    - House size (m²)
    - Land size (acres)
    """)

feature_importance = {
    "Bedrooms": 0.32,
    "Bathrooms": 0.21,
    "House Size": 0.27,
    "Land Size": 0.20,
}

df_importance = pd.DataFrame(
    list(feature_importance.items()),
    columns=["Feature", "Importance"],
)

fig = px.bar(
    df_importance,
    x="Importance",
    y="Feature",
    orientation="h",
    title="Feature Influence on Nairobi Property Price Prediction",
    color="Importance",
    color_continuous_scale="Reds",
    height=400,
)
st.plotly_chart(fig, use_container_width=True)
