import streamlit as st
import joblib
import pandas as pd
import requests
import json
from datetime import datetime
from math import ceil

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
#Hour,	Temperature,	Dew Point,	Relative Humidity,	Surface Albedo,	Pressure, and Wind Speed needed for prediction
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

# Placeholder for last prediction (GHI) so we can use it for panel calculations
predicted_ghi = None


# Run prediction only when user clicks the button
if st.button("Predict"):
    if scaler is None or knn is None:
        st.error("Model or scaler not loaded. Ensure `scaler_path` and `knn_path` are defined and files exist.")
    else:
        try:
            scaled_data = scaler.transform(df)
            scaled_data_df = pd.DataFrame(scaled_data, columns=features.keys())
            prediction = knn.predict(scaled_data_df)
            st.success(f"GHI (W/m²): {prediction[0]}")
            predicted_ghi = float(prediction[0])
            st.markdown("**Input meta-data**")
            # meta will be displayed after user-provided meta inputs below
        except Exception as e:
            st.error(f"Prediction failed: {e}")
st.write("## Now let's calculate the required panel surface area based on your inputs:")
# Additional (meta) inputs that are not part of the original model features
# Ask for required total power (watts) rather than a radiation value
required_power = st.number_input("Required Power (W)", min_value=0.0, max_value=1000000.0, value=800.0)

# Offer panel brand options (efficiency, typical area, nominal power). Users
# can pick a brand or choose "Custom" to enter their own efficiency and area.
st.subheader("Panel Brand / Efficiency Options")
brands = [
    {"Brand": "SunPower", "Efficiency (%)": 22.8, "Area (m^2)": 1.63, "Nominal Power (W)": 430},
    {"Brand": "LG", "Efficiency (%)": 20.4, "Area (m^2)": 1.70, "Nominal Power (W)": 400},
    {"Brand": "Panasonic", "Efficiency (%)": 20.3, "Area (m^2)": 1.70, "Nominal Power (W)": 405},
    {"Brand": "Jinko (Generic)", "Efficiency (%)": 18.4, "Area (m^2)": 1.70, "Nominal Power (W)": 375},
    {"Brand": "Custom", "Efficiency (%)": None, "Area (m^2)": None, "Nominal Power (W)": None}
]
brands_df = pd.DataFrame(brands)
st.table(brands_df)

brand_names = brands_df["Brand"].tolist()
chosen_brand = st.selectbox("Choose a panel brand (or Custom)", brand_names)

if chosen_brand == "Custom":
    panel_efficiency = st.number_input("Custom Panel Efficiency (%)", min_value=0.0, max_value=100.0, value=18.0)
    panel_area = st.number_input("Custom Panel Area (m^2)", min_value=0.1, max_value=10.0, value=1.7)
    nominal_power = st.number_input("Custom Panel Nominal Power (W)", min_value=1.0, max_value=2000.0, value=400.0)
else:
    row = brands_df[brands_df["Brand"] == chosen_brand].iloc[0]
    panel_efficiency = float(row["Efficiency (%)"])
    panel_area = float(row["Area (m^2)"])
    nominal_power = float(row["Nominal Power (W)"])

# System derate to account for system/inverter/losses
derate = st.slider("System derate (loss factor)", min_value=0.5, max_value=1.0, value=0.77)

# Display meta now (user inputs used for downstream calculations)
meta = {
    "Required Power (W)": required_power,
    "Panel Efficiency (%)": panel_efficiency,
    "Panel Area (m^2)": panel_area,
    "Panel Nominal Power (W)": nominal_power,
    "System Derate": derate
}

st.markdown("**Input meta-data**")
st.write(meta)

# Determine GHI to use for power calculation: prefer model prediction, otherwise ask user
if 'predicted_ghi' in globals() and predicted_ghi is not None:
    st.write(f"Predicted GHI (used for calculation): {predicted_ghi:.2f} W/m²")
    ghi_value = predicted_ghi
else:
    ghi_value = st.number_input("Expected GHI (W/m²) to use for calculation", min_value=0.0, value=800.0)

if st.button("Calculate Panels"):
    try:
        if ghi_value <= 0:
            st.error("GHI used for calculation must be greater than 0.")
        else:
            # Instantaneous power produced by one panel at given GHI
            power_per_panel = ghi_value * panel_area * (panel_efficiency / 100.0) * derate
            if power_per_panel <= 0:
                st.error("Computed power per panel is not positive. Check inputs.")
            else:
                panels_needed = ceil(required_power / power_per_panel)
                total_panel_area = panels_needed * panel_area
                estimated_system_power = panels_needed * power_per_panel

                st.success(f"Panels needed: {panels_needed}")
                st.write(f"Estimated instantaneous power from system: {estimated_system_power:.2f} W")
                st.write(f"Total panel area required: {total_panel_area:.2f} m²")
                st.write(f"Using panel nominal power ~ {nominal_power} W each, {ceil(required_power/nominal_power)} panels (approx) would be needed to meet required power at STC ratings.")
    except Exception as e:
        st.error(f"Panel calculation failed: {e}")