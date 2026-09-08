"""
Streamlit Dashboard for Neural Network House Price Prediction
"""
import streamlit as st
import requests
import pandas as pd
import plotly.express as px

st.set_page_config(
    page_title="Neural Network - House Price Predictor",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 Neural Network House Price Predictor")
st.markdown("**Model:** PyTorch Neural Network | **R² Score:** 0.8344")

# Sidebar inputs
st.sidebar.header("📊 Input Features")
st.sidebar.markdown("Adjust the sliders to predict house prices")

med_inc = st.sidebar.slider("Median Income", 0.0, 15.0, 3.5, 0.1)
house_age = st.sidebar.slider("House Age (years)", 0.0, 50.0, 20.0, 1.0)
ave_rooms = st.sidebar.slider("Average Rooms", 0.0, 10.0, 5.0, 0.1)
ave_bedrms = st.sidebar.slider("Average Bedrooms", 0.0, 5.0, 1.0, 0.1)
population = st.sidebar.slider("Population", 0.0, 50000.0, 1000.0, 100.0)
ave_occup = st.sidebar.slider("Average Occupancy", 0.0, 10.0, 3.0, 0.1)
latitude = st.sidebar.slider("Latitude", 32.0, 42.0, 37.0, 0.1)
longitude = st.sidebar.slider("Longitude", -125.0, -114.0, -122.0, 0.1)

# Prediction button
if st.sidebar.button("🔮 Predict Price", type="primary", use_container_width=True):
    data = {
        "MedInc": med_inc,
        "HouseAge": house_age,
        "AveRooms": ave_rooms,
        "AveBedrms": ave_bedrms,
        "Population": population,
        "AveOccup": ave_occup,
        "Latitude": latitude,
        "Longitude": longitude
    }
    
    try:
        response = requests.post("http://localhost:8000/predict", json=data)
        if response.status_code == 200:
            result = response.json()
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    "🏠 Predicted Price",
                    f"${result['prediction']:.2f}K",
                    delta="Neural Network Prediction"
                )
            
            with col2:
                st.metric(
                    "📈 Model R²",
                    "0.8344",
                    delta="+0.0082 vs XGBoost"
                )
            
            with col3:
                st.metric(
                    "🎯 Accuracy",
                    "± $0.46K",
                    delta="Average Error"
                )
            
            # Show input summary
            st.subheader("📊 Input Features")
            df_input = pd.DataFrame([data])
            st.dataframe(df_input, use_container_width=True)
        else:
            st.error(f"API Error: {response.status_code}")
    except Exception as e:
        st.error(f"Connection error: {e}")

# Main content
st.subheader("🧠 Neural Network Architecture")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    **Architecture:**
    - Input Layer: 8 features
    - Hidden Layer 1: 64 neurons (ReLU + Dropout)
    - Hidden Layer 2: 32 neurons (ReLU + Dropout)
    - Hidden Layer 3: 16 neurons (ReLU + Dropout)
    - Output Layer: 1 neuron
    
    **Total Parameters:** 3,201
    """)

with col2:
    st.markdown("""
    **Model Performance:**
    - R² Score: **0.8344** (Best!)
    - RMSE: **$0.46K**
    - Training Loss: 0.0955
    - Test Loss: 0.1040
    
    **🏆 This model beats XGBoost!**
    """)

# Feature importance (approximate from first layer weights)
st.subheader("🔍 Feature Influence")

feature_importance = {
    "MedInc": 0.32,
    "Latitude": 0.19,
    "Longitude": 0.16,
    "AveOccup": 0.13,
    "HouseAge": 0.09,
    "AveRooms": 0.05,
    "Population": 0.04,
    "AveBedrms": 0.02
}

df_importance = pd.DataFrame(
    list(feature_importance.items()),
    columns=["Feature", "Importance"]
)

fig = px.bar(
    df_importance,
    x="Importance",
    y="Feature",
    orientation='h',
    title="Feature Influence on Neural Network Predictions",
    color="Importance",
    color_continuous_scale="Reds",
    height=400
)
st.plotly_chart(fig, use_container_width=True)
