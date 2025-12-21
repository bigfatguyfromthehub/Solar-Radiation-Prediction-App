import streamlit as st
import joblib
import pandas as pd

st.title("Solar Radiation Prediction")
knn_path = "knn.pkl"  # Path to the trained KNN.pkl model
scaler_path = "Modelsscaler.pkl"   # Path to the scaler used during model training
# Attempt to load scaler and model, show error in UI if loading fails
try:
    scaler = joblib.load(scaler_path)
    knn = joblib.load(knn_path)
except Exception as e:
    st.error(f"Failed to load model or scaler: {e}")
    scaler = None
    knn = None

# Create input fields for user input
st.subheader("Enter Features")
col1, col2 = st.columns(2)

with col1:
    hour = st.number_input("Hour", min_value=0, max_value=23, value=16)
    temperature = st.number_input("Temperature (°C)", min_value=-50.0, max_value=60.0, value=29.6)
    dew_point = st.number_input("Dew Point (°C)", min_value=-50.0, max_value=40.0, value=16.8)
    relative_humidity = st.number_input("Relative Humidity (%)", min_value=0.0, max_value=100.0, value=46.17)

with col2:
    surface_albedo = st.number_input("Surface Albedo", min_value=0.0, max_value=1.0, value=0.15)
    pressure = st.number_input("Pressure (hPa)", min_value=800, max_value=1100, value=986)
    wind_speed = st.number_input("Wind Speed (m/s)", min_value=0.0, max_value=50.0, value=2.9)

# Additional (meta) inputs that are not part of the original model features
required_radiation = st.number_input("Required Radiation (W/m²)", min_value=0.0, max_value=2000.0, value=800.0)
panel_efficiency = st.number_input("Panel Efficiency (%)", min_value=0.0, max_value=100.0, value=18.0)

# Build feature dataframe for the model (keep only the features the model expects)
features = {
    "Hour": hour,
    "Temperature": temperature,
    "Dew Point": dew_point,
    "Relative Humidity": relative_humidity,
    "Surface Albedo": surface_albedo,
    "Pressure": pressure,
    "Wind Speed": wind_speed
}
df = pd.DataFrame(features, index=[0])

# Meta data (shown to user / used for downstream calculations but NOT passed
# through the scaler/model unless the model was trained with them)
meta = {
    "Required Radiation (W/m²)": required_radiation,
    "Panel Efficiency (%)": panel_efficiency
}

# Run prediction only when user clicks the button
if st.button("Predict"):
    if scaler is None or knn is None:
        st.error("Model or scaler not loaded. Ensure `scaler_path` and `knn_path` are defined and files exist.")
    else:
        try:
            scaled_data = scaler.transform(df)
            scaled_data_df = pd.DataFrame(scaled_data, columns=features.keys())
            prediction = knn.predict(scaled_data_df)
            st.success(f"Prediction: {prediction[0]}")
            st.markdown("**Input meta-data**")
            st.write(meta)
        except Exception as e:
            st.error(f"Prediction failed: {e}")