import streamlit as st
import joblib
import pandas as pd
import requests
import json
from datetime import datetime

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

lat=37.45980962438753
lon=-122.1511311602308

city = st.text_input("City", value="Menlo Park")
def get_coordinates(city_name):
       api_key = "4e514b49d73362c5d739f05fea7f27cd" # This API key is for geocoding
       url = f"http://api.openweathermap.org/geo/1.0/direct?q={city_name}&limit=1&appid={api_key}"
       try:
              response = requests.get(url).json()

              # Check if the response is a list and contains elements
              if isinstance(response, list) and len(response) > 0:
                     lat = response[0]['lat']
                     lon = response[0]['lon']
                     print(f"City: {city_name}") # Using city_name parameter
                     print(f"Latitude: {lat}, Longitude: {lon}")
                     return lat, lon
              else:
                     # Handle cases where response is an empty list or an error dictionary
                     print(f"City '{city_name}' not found!")
                     if isinstance(response, dict) and 'message' in response:
                            print(f"API Error from geocoding: {response['message']}")
                     return None, None
       except requests.exceptions.RequestException as e:
              print(f"Error making geocoding API request: {e}")
              return None, None
       except json.JSONDecodeError:
              print("Error decoding JSON response from geocoding API.")
              return None, None
       except Exception as e:
              print(f"An unexpected error occurred in get_coordinates: {e}")
              return None, None

lat, lon = get_coordinates(city)

# Ensure these exist even if coordinates lookup fails
weather_data = None
uvi_data = None

# Only proceed if valid coordinates were obtained
try:
    if lat is not None and lon is not None:
        # Fetch weather data from OpenWeatherMap API
        # Note: The appid for weather data (4e514b49d73362c5d739f05fea7f27cd) is different
        weather_api_url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid=4e514b49d73362c5d739f05fea7f27cd"
        weather_data = None
        uvi_data = None
        uvi_url = f"https://api.openweathermap.org/data/2.5/uvi?lat={lat}&lon={lon}&appid=4e514b49d73362c5d739f05fea7f27cd" # The original code defines uvi_url but doesn't use it.

        try:
                response = requests.get(weather_api_url)
                if response.status_code == 200:
                        weather_data = response.json()
                        print(weather_data)
                else:
                        print(f"Failed to fetch weather data. Status code: {response.status_code}")
                        if response.text:
                                print(f"Response content: {response.text}")
        except Exception as e:
                weather_data = None
                print(f"An error occurred while fetching weather data: {e}")
    else:
        print("Weather data and UVI data not fetched due to invalid coordinates.")
    st.write(f"Fetched current weather data for {city}")
except Exception as e:
    st.write("Using the default weather data due to an error:", e)

# Default values in case API call fails
default_humidity = 46
default_pressure = 986
default_temperature = 29.6
default_wind_speed = 2.9
if weather_data:
    default_humidity = weather_data.get('main', {}).get('humidity', default_humidity)
    default_pressure = weather_data.get('main', {}).get('pressure', default_pressure)
    # Convert Kelvin to Fahrenheit for temperature
    kelvin_temp = weather_data.get('main', {}).get('temp', 302.15)
    default_temperature = kelvin_temp - 273.15
    default_wind_speed = weather_data.get('wind', {}).get('speed', default_wind_speed)
default_dew_point = default_temperature - ((100 - default_humidity) / 5.)
default_hour = datetime.now().hour
# Create input fields for user input
st.subheader("Enter Features")
col1, col2 = st.columns(2)

with col1:
    hour = st.number_input("Hour", min_value=0, max_value=23, value=default_hour)
    temperature = st.number_input("Temperature (°C)", min_value=-50.0, max_value=60.0, value=default_temperature)
    dew_point = st.number_input("Dew Point (°C)", min_value=-50.0, max_value=40.0, value=default_dew_point)
    relative_humidity = st.number_input("Relative Humidity (%)", min_value=0, max_value=100, value=default_humidity)

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