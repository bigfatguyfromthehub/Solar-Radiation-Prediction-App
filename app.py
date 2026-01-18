import streamlit as st
import joblib
import pandas as pd
import requests
import json
from datetime import datetime
from math import ceil

# Configure page
st.set_page_config(
    page_title="Solar Radiation Prediction App",
    page_icon="☀️",
    layout="wide"
)

# Load model globally (used by multiple pages)
knn_path = "knn.pkl"  # Path to the trained KNN.pkl model
try:
    knn = joblib.load(knn_path)
except Exception as e:
    knn = None
    st.error(f"Failed to load model: {e}")

# Helper functions
def get_coordinates(city_name):
    """Fetch latitude and longitude for a given city using OpenWeatherMap API."""
    api_key = "4e514b49d73362c5d739f05fea7f27cd"
    url = f"http://api.openweathermap.org/geo/1.0/direct?q={city_name}&limit=1&appid={api_key}"
    try:
        response = requests.get(url).json()
        if isinstance(response, list) and len(response) > 0:
            lat = response[0]['lat']
            lon = response[0]['lon']
            return lat, lon
        else:
            return None, None
    except Exception as e:
        return None, None

def fetch_weather_data(lat, lon, city_name):
    """Fetch weather data for the given coordinates."""
    api_key = "4e514b49d73362c5d739f05fea7f27cd"
    weather_api_url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={api_key}"
    
    try:
        response = requests.get(weather_api_url)
        if response.status_code == 200:
            return response.json()
    except Exception as e:
        pass
    return None


# ============== PAGE: HOME ==============
def page_home():
    st.title("☀️ Solar Radiation Prediction App")
    
    st.markdown("---")
    st.header("Welcome to the Solar Radiation Prediction Application")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ### 📋 Project Overview
        
        The **Solar Radiation Prediction App** is designed to help you estimate solar radiation levels 
        and calculate the solar panel requirements needed to meet your energy needs. This tool combines 
        weather data and machine learning predictions to provide accurate solar energy assessments.
        
        ### 🎯 Objectives
        
        1. **Predict Solar Radiation**: Use a KNN machine learning model to predict Global Horizontal 
           Irradiance (GHI) based on weather parameters
        2. **Calculate Panel Requirements**: Determine how many solar panels you need to meet your 
           power requirements
        3. **Estimate Energy Generation**: Project annual energy production and potential cost savings
        4. **Location-Based Analysis**: Get weather data automatically for any city worldwide
        
        ### 🚀 How to Use the Calculator
        
        **Step 1: Navigate to the Prediction Page**
        - Go to the "Prediction" page using the sidebar menu
        
        **Step 2: Enter Location Information**
        - Enter a city name (e.g., "Menlo Park") to automatically fetch current weather data
        - The app uses this data to pre-fill weather parameters
        
        **Step 3: Predict Solar Radiation**
        - Review and adjust weather parameters (temperature, humidity, pressure, wind speed, etc.)
        - Click "Predict" to calculate the GHI (Global Horizontal Irradiance) using our ML model
        
        **Step 4: Calculate Panel Requirements**
        - Enter your required power output in Watts
        - Select a solar panel brand or enter custom specifications
        - Adjust the system derate factor to account for real-world losses
        - Click "Calculate Panels" to see how many panels you need
        
        **Step 5: Review Annual Projections**
        - View estimated annual energy generation in kWh
        - Calculate potential annual savings based on local electricity rates
        """)
    
    with col2:
        st.info("""
        ### ⚡ Key Metrics
        
        **What We Calculate:**
        - Global Horizontal Irradiance (GHI)
        - Number of panels needed
        - System instantaneous power
        - Annual energy production
        - Estimated annual savings
        
        **Input Parameters:**
        - Hour of day
        - Temperature
        - Humidity
        - Pressure
        - Wind speed
        - Panel efficiency
        - System losses
        """)
    
    st.markdown("---")
    st.subheader("📊 What is Solar Radiation?")
    st.markdown("""
    **Global Horizontal Irradiance (GHI)** is the total solar radiation received on a horizontal surface. 
    It's measured in watts per square meter (W/m²) and varies based on:
    - Time of day and season
    - Weather conditions (clouds, humidity)
    - Atmospheric pressure
    - Surface albedo (reflectivity)
    - Wind speed effects on atmospheric clarity
    
    Our machine learning model predicts GHI based on these weather parameters, allowing you to estimate 
    solar panel performance at any time and location.
    """)
    
    st.markdown("---")
    st.subheader("🔧 About Solar Panel Calculations")
    st.markdown("""
    The app calculates panel requirements using the formula:
    
    **Power per panel = GHI × Panel Area × Efficiency × System Derate**
    
    - **GHI**: Predicted solar radiation (W/m²)
    - **Panel Area**: Physical size of each panel (m²)
    - **Efficiency**: How well the panel converts sunlight to electricity (%)
    - **System Derate**: Factor accounting for real-world losses (~0.77 typical)
    
    **Annual Energy** is calculated by multiplying daily energy by peak sun hours and 365 days.
    """)
    
    st.markdown("---")
    st.subheader("📍 Common Peak Sun Hours by Location")
    peak_hours_data = {
        "Location": ["Northern California (Menlo Park)", "Southern California (Los Angeles)", 
                     "Arizona (Phoenix)", "Texas (Houston)", "Florida (Miami)", "New York (NYC)"],
        "Avg Peak Hours/Day": [5.0, 5.5, 6.0, 5.0, 4.5, 4.0]
    }
    st.dataframe(pd.DataFrame(peak_hours_data), use_container_width=True)


# ============== PAGE: PREDICTION ==============
def page_prediction():
    st.title("📈 Solar Radiation Prediction")
    
    # Input city and fetch weather
    city = st.text_input("Enter City Name", value="Menlo Park", key="city_input")
    
    lat, lon = get_coordinates(city)
    
    if lat is None or lon is None:
        st.warning(f"Could not find coordinates for '{city}'. Using default values.")
        lat, lon = 37.45980962438753, -122.1511311602308
    else:
        st.success(f"✓ Found {city} (Lat: {lat:.2f}, Lon: {lon:.2f})")
    
    # Fetch weather data
    weather_data = fetch_weather_data(lat, lon, city)
    
    # Set defaults
    default_humidity = 46
    default_pressure = 986
    default_temperature = 29.6
    default_wind_speed = 2.9
    
    if weather_data:
        default_humidity = weather_data.get('main', {}).get('humidity', default_humidity)
        default_pressure = weather_data.get('main', {}).get('pressure', default_pressure)
        kelvin_temp = weather_data.get('main', {}).get('temp', 302.15)
        default_temperature = kelvin_temp - 273.15
        default_wind_speed = weather_data.get('wind', {}).get('speed', default_wind_speed)
        st.info(f"✓ Fetched current weather for {city}")
    
    default_dew_point = default_temperature - ((100 - default_humidity) / 5.)
    default_hour = datetime.now().hour
    
    # Create input fields
    st.subheader("Enter Weather Features")
    col1, col2 = st.columns(2)
    
    with col1:
        hour = st.number_input("Hour (0-23)", min_value=0, max_value=23, value=default_hour, key="hour")
        temperature = st.number_input("Temperature (°C)", min_value=-50.0, max_value=60.0, 
                                     value=default_temperature, key="temp")
        dew_point = st.number_input("Dew Point (°C)", min_value=-50.0, max_value=40.0, 
                                   value=default_dew_point, key="dew")
        relative_humidity = st.number_input("Relative Humidity (%)", min_value=0, max_value=100, 
                                           value=default_humidity, key="humidity")
    
    with col2:
        surface_albedo = st.number_input("Surface Albedo", min_value=0.0, max_value=1.0, 
                                        value=0.15, key="albedo")
        pressure = st.number_input("Pressure (hPa)", min_value=800, max_value=1100, 
                                  value=default_pressure, key="pressure")
        wind_speed = st.number_input("Wind Speed (m/s)", min_value=0.0, max_value=50.0, 
                                    value=default_wind_speed, key="wind")
    
    # Build feature dataframe
    feature_names = ['Hour', 'Temperature', 'Dew Point', 'Relative Humidity', 'Surface Albedo', 'Pressure', 'Wind Speed']
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
    
    # Initialize session state
    if "predicted_ghi" not in st.session_state:
        st.session_state.predicted_ghi = None
    
    # Prediction button
    if st.button("🔮 Predict GHI", key="predict_btn"):
        if knn is None:
            st.error("Model not loaded. Ensure `knn.pkl` exists.")
        else:
            try:
                scaled_data = df[feature_names]
                prediction = knn.predict(scaled_data)
                st.session_state.predicted_ghi = float(prediction[0])
                st.success(f"✓ Predicted GHI: **{st.session_state.predicted_ghi:.2f} W/m²**")
            except Exception as e:
                st.error(f"Prediction failed: {e}")
    
    st.divider()
    
    # Display prediction status
    st.subheader("Prediction Status")
    if st.session_state.predicted_ghi is None:
        st.info("👆 Click 'Predict GHI' above to generate a prediction before proceeding to panel calculations.")
    else:
        st.success(f"✓ Current GHI Prediction: **{st.session_state.predicted_ghi:.2f} W/m²**")
        st.caption("Use this value to calculate panel requirements")


# ============== PAGE: CALCULATOR ==============
def page_calculator():
    st.title("🔧 Solar Panel Calculator")
    
    # Check if prediction exists
    if "predicted_ghi" not in st.session_state or st.session_state.predicted_ghi is None:
        st.warning("⚠️ No GHI prediction available. Please go to the **Prediction** page and click 'Predict GHI' first.")
        st.info("The prediction is needed to calculate panel requirements accurately.")
        return
    
    st.success(f"✓ Using predicted GHI: **{st.session_state.predicted_ghi:.2f} W/m²**")
    
    st.subheader("System Requirements")
    required_power = st.number_input("Required Power Output (W)", min_value=0.0, max_value=1000000.0, 
                                     value=800.0, key="required_power")
    
    # Panel brand options
    st.subheader("Select Solar Panel")
    brands = [
        {"Brand": "SunPower", "Efficiency (%)": 22.8, "Area (m²)": 1.63, "Nominal Power (W)": 430},
        {"Brand": "LG", "Efficiency (%)": 20.4, "Area (m²)": 1.70, "Nominal Power (W)": 400},
        {"Brand": "Panasonic", "Efficiency (%)": 20.3, "Area (m²)": 1.70, "Nominal Power (W)": 405},
        {"Brand": "Jinko (Generic)", "Efficiency (%)": 18.4, "Area (m²)": 1.70, "Nominal Power (W)": 375},
        {"Brand": "Custom", "Efficiency (%)": None, "Area (m²)": None, "Nominal Power (W)": None}
    ]
    brands_df = pd.DataFrame(brands)
    
    with st.expander("View Panel Specifications", expanded=False):
        st.table(brands_df)
    
    chosen_brand = st.selectbox("Choose a panel brand", [b["Brand"] for b in brands], key="brand_select")
    
    if chosen_brand == "Custom":
        col1, col2, col3 = st.columns(3)
        with col1:
            panel_efficiency = st.number_input("Efficiency (%)", min_value=0.0, max_value=100.0, 
                                              value=18.0, key="custom_eff")
        with col2:
            panel_area = st.number_input("Area (m²)", min_value=0.1, max_value=10.0, 
                                        value=1.7, key="custom_area")
        with col3:
            nominal_power = st.number_input("Nominal Power (W)", min_value=1.0, max_value=2000.0, 
                                           value=400.0, key="custom_power")
    else:
        row = brands_df[brands_df["Brand"] == chosen_brand].iloc[0]
        panel_efficiency = float(row["Efficiency (%)"])
        panel_area = float(row["Area (m²)"])
        nominal_power = float(row["Nominal Power (W)"])
    
    # System derate
    derate = st.slider("System Derate Factor (accounts for losses)", min_value=0.5, max_value=1.0, 
                       value=0.77, step=0.01, key="derate")
    
    # Override GHI option
    override_ghi = st.checkbox("Override GHI for testing scenarios?", key="override_ghi")
    if override_ghi:
        ghi_value = st.number_input("Override GHI (W/m²)", min_value=0.0, value=800.0, key="override_ghi_val")
    else:
        ghi_value = st.session_state.predicted_ghi
    
    # Peak sun hours
    st.subheader("Location Settings")
    peak_sun_hours = st.slider("Average Peak Sun Hours per Day", min_value=2.0, max_value=8.0, 
                               value=5.0, step=0.5, key="peak_hours",
                               help="Typical solar insolation hours. Menlo Park avg: ~5 hours/day")
    
    # Calculate button
    if st.button("📊 Calculate Panels", key="calculate_btn"):
        if ghi_value is None or ghi_value <= 0:
            st.error("GHI value must be greater than 0.")
        else:
            try:
                # Power per panel
                power_per_panel = ghi_value * panel_area * (panel_efficiency / 100.0) * derate
                
                if power_per_panel <= 0:
                    st.error("Computed power per panel is not positive. Check inputs.")
                else:
                    panels_needed = ceil(required_power / power_per_panel)
                    total_panel_area = panels_needed * panel_area
                    estimated_system_power = panels_needed * power_per_panel
                    
                    # Results
                    st.divider()
                    st.subheader("📊 Results")
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Panels Needed", panels_needed)
                    with col2:
                        st.metric("Total Panel Area", f"{total_panel_area:.2f} m²")
                    with col3:
                        st.metric("System Power", f"{estimated_system_power:.0f} W")
                    
                    st.markdown("**Calculation Details:**")
                    st.write(f"- Power per panel at {ghi_value:.0f} W/m²: **{power_per_panel:.2f} W**")
                    st.write(f"- Total panels needed: **{panels_needed}**")
                    st.write(f"- Total system area: **{total_panel_area:.2f} m²**")
                    st.write(f"- Estimated system power: **{estimated_system_power:.2f} W**")
                    
                    # Annual energy
                    st.divider()
                    st.subheader("📈 Annual Energy Generation")
                    
                    energy_per_panel_year = (power_per_panel / 1000) * peak_sun_hours * 365
                    total_system_energy = energy_per_panel_year * panels_needed
                    
                    st.metric("Annual System Energy", f"{total_system_energy:,.0f} kWh/year")
                    
                    st.markdown("**Energy Breakdown:**")
                    st.write(f"- Energy per panel per year: {energy_per_panel_year:,.0f} kWh/year")
                    st.write(f"- Total panels: {panels_needed}")
                    st.write(f"- Peak sun hours/day: {peak_sun_hours}")
                    
                    # Savings
                    st.divider()
                    st.subheader("💰 Financial Analysis")
                    
                    avg_electricity_rate = st.number_input("Local electricity rate ($/kWh)", 
                                                           min_value=0.0, value=0.12, key="electricity_rate")
                    annual_savings = total_system_energy * avg_electricity_rate
                    
                    st.metric("Estimated Annual Savings", f"${annual_savings:,.2f}", 
                             delta=f"at ${avg_electricity_rate}/kWh")
                    
                    # Payback period estimate
                    system_cost = st.number_input("Estimated system installation cost ($)", 
                                                 min_value=0.0, value=8000.0, key="system_cost")
                    
                    if system_cost > 0 and annual_savings > 0:
                        payback_years = system_cost / annual_savings
                        st.write(f"**Estimated Payback Period: {payback_years:.1f} years**")
                    
            except Exception as e:
                st.error(f"Calculation failed: {e}")


# ============== PAGE: ABOUT ==============
def page_about():
    st.title("ℹ️ About This Application")
    
    st.markdown("---")
    st.header("Application Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🛠️ Technology Stack")
        st.markdown("""
        - **Frontend**: Streamlit (Python web framework)
        - **ML Model**: K-Nearest Neighbors (KNN)
        - **Weather Data**: OpenWeatherMap API
        - **Data Processing**: Pandas, NumPy
        - **Visualization**: Streamlit charts and metrics
        """)
        
        st.subheader("📊 Model Details")
        st.markdown("""
        - **Algorithm**: K-Nearest Neighbors (KNN)
        - **Input Features**: 7 weather parameters
        - **Target Variable**: Global Horizontal Irradiance (GHI)
        - **Pre-training**: Normalized input features
        - **Output**: Solar radiation prediction in W/m²
        """)
    
    with col2:
        st.subheader("🌍 Data Sources")
        st.markdown("""
        - **Weather Data**: OpenWeatherMap API
          - Current conditions for any location
          - Temperature, humidity, pressure, wind speed
        
        - **Solar Data**: Training model based on 
          real solar measurement datasets
        
        - **Location Data**: Geographic coordinate lookup
        """)
        
        st.subheader("✨ Features")
        st.markdown("""
        ✓ Real-time weather integration
        ✓ ML-based solar radiation prediction
        ✓ Customizable panel specifications
        ✓ Annual energy generation estimates
        ✓ Financial savings calculations
        ✓ Location-based analysis
        """)
    
    st.markdown("---")
    st.subheader("📚 How Solar Radiation Prediction Works")
    
    st.markdown("""
    ### The Machine Learning Approach
    
    Our KNN model predicts solar radiation by analyzing 7 key weather parameters:
    
    1. **Hour** (0-23): Captures daily solar cycle
    2. **Temperature (°C)**: Indicates atmospheric conditions
    3. **Dew Point (°C)**: Measures atmospheric moisture
    4. **Relative Humidity (%)**: Affects cloud formation and clarity
    5. **Surface Albedo**: Ground reflectivity affecting radiation
    6. **Pressure (hPa)**: Indicates atmospheric density
    7. **Wind Speed (m/s)**: Affects aerosol concentration and clarity
    
    The model finds the K nearest neighbors in the training data with similar weather conditions,
    then averages their GHI values to make a prediction.
    
    ### Why These Parameters?
    
    - **Hour & Temperature**: Solar angle and intensity vary throughout the day and seasons
    - **Humidity & Dew Point**: High moisture increases cloud likelihood, reducing radiation
    - **Pressure**: Lower pressure often correlates with weather patterns affecting solar radiation
    - **Wind Speed**: Clears aerosols, typically increasing radiation; very high wind suggests storms
    - **Albedo**: Affects how much radiation is reflected vs. absorbed by the surface
    """)
    
    st.markdown("---")
    st.subheader("🔬 Solar Panel Efficiency Factors")
    
    st.markdown("""
    ### Why System Derate?
    
    Real-world solar systems don't achieve 100% efficiency. The "derate factor" (typically 0.77 or 77%)
    accounts for:
    
    | Factor | Loss | Impact |
    |--------|------|--------|
    | Inverter losses | ~3-5% | DC to AC conversion inefficiency |
    | Wiring losses | ~2-3% | Resistance in electrical connections |
    | Temperature effects | ~15-20% | Panels less efficient when hot |
    | Soiling/dirt | ~2-5% | Dust and debris on panels |
    | Mismatch losses | ~2% | Variation between panels |
    | Tracking losses | ~5-10% | If not using sun-tracking (fixed systems) |
    
    **Total typical derate: ~23-30% (leaving 70-77% efficiency)**
    
    ### Panel Efficiency Ratings
    
    Panel efficiency varies by brand and technology:
    - **Standard monocrystalline**: 18-20%
    - **High-efficiency monocrystalline**: 21-23% (e.g., SunPower)
    - **Polycrystalline**: 15-17%
    
    These are rated under Standard Test Conditions (STC):
    - Irradiance: 1000 W/m²
    - Temperature: 25°C
    - Air mass: 1.5
    """)
    
    st.markdown("---")
    st.subheader("❓ FAQ")
    
    with st.expander("What is GHI (Global Horizontal Irradiance)?"):
        st.markdown("""
        GHI is the total solar radiation (direct + diffuse) hitting a horizontal surface on the ground.
        It's measured in watts per square meter (W/m²) and is the primary metric for ground-based
        solar systems. It varies by location, time, weather, and season.
        """)
    
    with st.expander("Why does my prediction vary each time?"):
        st.markdown("""
        The KNN model's prediction depends on the weather parameters you input. If you change any
        weather value (temperature, humidity, time, etc.), the prediction will change accordingly.
        To get consistent predictions, use the same location and weather parameters.
        """)
    
    with st.expander("How accurate is this model?"):
        st.markdown("""
        The model's accuracy depends on the training data quality and the accuracy of weather data.
        Predictions are best during clear weather conditions. Cloudy and extreme weather conditions
        may have higher uncertainty. Use the override GHI feature to test different scenarios.
        """)
    
    with st.expander("Can I use this for my roof?"):
        st.markdown("""
        Yes! This calculator helps estimate:
        - How much solar radiation your location receives
        - How many panels you need for your power requirements
        - Expected annual energy generation
        - Potential savings
        
        However, consult with professional installers regarding:
        - Roof structural capacity
        - Optimal panel orientation and tilt
        - Local building codes and permits
        - Interconnection requirements
        """)
    
    with st.expander("What about seasonal variations?"):
        st.markdown("""
        Solar radiation varies significantly by season due to:
        - Solar angle (affects time of day GHI peaks)
        - Day length (winter days are shorter)
        - Weather patterns (more cloud cover in some seasons)
        
        Use the "Average Peak Sun Hours" slider to adjust for your location's seasonal average.
        Most calculators assume an annual average; real systems will produce more in summer
        and less in winter.
        """)
    
    st.markdown("---")
    st.subheader("📞 Support & Feedback")
    st.markdown("""
    For questions, issues, or suggestions about this application, please refer to the project documentation
    or contact the development team.
    
    **Version**: 2.0 (Multi-page interface)  
    **Last Updated**: January 2026
    """)


# ============== MAIN APP NAVIGATION ==============
def main():
    st.sidebar.title("🌞 Navigation")
    
    page_options = {
        "🏠 Home": page_home,
        "📈 Prediction": page_prediction,
        "🔧 Calculator": page_calculator,
        "ℹ️ About": page_about,
    }
    
    selected_page = st.sidebar.radio(
        "Select a page:",
        list(page_options.keys()),
        index=0
    )
    
    st.sidebar.markdown("---")
    st.sidebar.caption("Solar Radiation Prediction App v2.0")
    
    # Run the selected page
    page_options[selected_page]()


if __name__ == "__main__":
    main()