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
    layout="wide"
)

# Global CSS — hide the Streamlit toolbar and tighten spacing so more
# content fits on a single screen without scrolling.
st.markdown("""
<style>
    /* Hide top toolbar (Deploy button, hamburger, etc.) */
    header[data-testid="stHeader"] { display: none !important; }

    /* Reduce main block top padding from ~6rem to 1rem */
    .block-container { padding-top: 1rem !important; padding-bottom: 0.5rem !important; }

    /* Shrink default h1 (st.title) size */
    h1 { font-size: 1.6rem !important; margin-bottom: 0.25rem !important; }

    /* Shrink h2/h3 (st.header / st.subheader) */
    h2 { font-size: 1.2rem !important; margin-bottom: 0.2rem !important; }
    h3 { font-size: 1.05rem !important; margin-bottom: 0.15rem !important; }

    /* Tighten spacing around metrics and widgets */
    div[data-testid="stMetric"]  { padding: 0.3rem 0 !important; }
    div[data-testid="stVerticalBlock"] > div { gap: 0.4rem !important; }
</style>
""", unsafe_allow_html=True)

# Load model globally (used by multiple pages)
knn_path = "knn.pkl"  # Path to the trained KNN.pkl model
try:
    knn = joblib.load(knn_path)
except Exception as e:
    knn = None
    st.error(f"Failed to load model: {e}")

# Average retail electricity rates ($/kWh) by country — IEA / GlobalPetrolPrices 2024
ELECTRICITY_RATES = {
    "Japan": 0.24, "India": 0.08, "China": 0.08, "Brazil": 0.14,
    "Mexico": 0.10, "Egypt": 0.04, "Pakistan": 0.09, "Argentina": 0.06,
    "Philippines": 0.18, "Nigeria": 0.05, "USA": 0.16, "UK": 0.34,
    "France": 0.23, "Germany": 0.40, "Russia": 0.06, "Thailand": 0.12,
    "Turkey": 0.11, "Canada": 0.13, "Australia": 0.25, "UAE": 0.08,
    "Singapore": 0.22, "Hong Kong": 0.16, "South Korea": 0.11,
    "Spain": 0.28, "Italy": 0.30, "Netherlands": 0.35, "South Africa": 0.14,
}
DEFAULT_RATE = 0.15  # fallback when country unknown

# City database with population data (sorted by population)
MAJOR_CITIES = [
    {"name": "Tokyo",           "lat": 35.6762,  "lon": 139.6503,  "population": 37400068, "country": "Japan"},
    {"name": "Delhi",           "lat": 28.7041,  "lon": 77.1025,   "population": 32941000, "country": "India"},
    {"name": "Shanghai",        "lat": 31.2304,  "lon": 121.4737,  "population": 27058000, "country": "China"},
    {"name": "São Paulo",       "lat": -23.5505, "lon": -46.6333,  "population": 22043028, "country": "Brazil"},
    {"name": "Mexico City",     "lat": 19.4326,  "lon": -99.1332,  "population": 21581000, "country": "Mexico"},
    {"name": "Cairo",           "lat": 30.0444,  "lon": 31.2357,   "population": 21323000, "country": "Egypt"},
    {"name": "Mumbai",          "lat": 19.0760,  "lon": 72.8777,   "population": 20961000, "country": "India"},
    {"name": "Beijing",         "lat": 39.9042,  "lon": 116.4074,  "population": 21540000, "country": "China"},
    {"name": "Osaka",           "lat": 34.6937,  "lon": 135.5023,  "population": 19223000, "country": "Japan"},
    {"name": "New York",        "lat": 40.7128,  "lon": -74.0060,  "population": 18823000, "country": "USA"},
    {"name": "Karachi",         "lat": 24.8607,  "lon": 67.0011,   "population": 15400000, "country": "Pakistan"},
    {"name": "Buenos Aires",    "lat": -34.6037, "lon": -58.3816,  "population": 15042000, "country": "Argentina"},
    {"name": "Los Angeles",     "lat": 34.0522,  "lon": -118.2437, "population": 13873000, "country": "USA"},
    {"name": "Manila",          "lat": 14.5994,  "lon": 120.9842,  "population": 13923000, "country": "Philippines"},
    {"name": "Kolkata",         "lat": 22.5726,  "lon": 88.3639,   "population": 14681000, "country": "India"},
    {"name": "Lagos",           "lat": 6.5244,   "lon": 3.3792,    "population": 13463000, "country": "Nigeria"},
    {"name": "Rio de Janeiro",  "lat": -22.9068, "lon": -43.1729,  "population": 12280000, "country": "Brazil"},
    {"name": "Guangzhou",       "lat": 23.1291,  "lon": 113.2644,  "population": 15301000, "country": "China"},
    {"name": "London",          "lat": 51.5074,  "lon": -0.1278,   "population": 9002488,  "country": "UK"},
    {"name": "Moscow",          "lat": 55.7558,  "lon": 37.6173,   "population": 12655050, "country": "Russia"},
    {"name": "Bangkok",         "lat": 13.7563,  "lon": 100.5018,  "population": 10899698, "country": "Thailand"},
    {"name": "Istanbul",        "lat": 41.0082,  "lon": 28.9784,   "population": 15029231, "country": "Turkey"},
    {"name": "Paris",           "lat": 48.8566,  "lon": 2.3522,    "population": 2161000,  "country": "France"},
    {"name": "Menlo Park",      "lat": 37.4530,  "lon": -122.1817, "population": 32026,    "country": "USA"},
    {"name": "San Francisco",   "lat": 37.7749,  "lon": -122.4194, "population": 873965,   "country": "USA"},
    {"name": "Chicago",         "lat": 41.8781,  "lon": -87.6298,  "population": 2716000,  "country": "USA"},
    {"name": "Houston",         "lat": 29.7604,  "lon": -95.3698,  "population": 2320268,  "country": "USA"},
    {"name": "Phoenix",         "lat": 33.4484,  "lon": -112.0742, "population": 1580619,  "country": "USA"},
    {"name": "Miami",           "lat": 25.7617,  "lon": -80.1918,  "population": 467963,   "country": "USA"},
    {"name": "Toronto",         "lat": 43.6532,  "lon": -79.3832,  "population": 2930000,  "country": "Canada"},
    {"name": "Mexico",          "lat": 19.4326,  "lon": -99.1332,  "population": 9209944,  "country": "Mexico"},
    {"name": "Berlin",          "lat": 52.5200,  "lon": 13.4050,   "population": 3645000,  "country": "Germany"},
    {"name": "Sydney",          "lat": -33.8688, "lon": 151.2093,  "population": 5312000,  "country": "Australia"},
    {"name": "Dubai",           "lat": 25.2048,  "lon": 55.2708,   "population": 3693000,  "country": "UAE"},
    {"name": "Singapore",       "lat": 1.3521,   "lon": 103.8198,  "population": 5638000,  "country": "Singapore"},
    {"name": "Hong Kong",       "lat": 22.3193,  "lon": 114.1694,  "population": 7500700,  "country": "Hong Kong"},
    {"name": "Seoul",           "lat": 37.5665,  "lon": 126.9780,  "population": 9776000,  "country": "South Korea"},
    {"name": "Bangkok",         "lat": 13.7563,  "lon": 100.5018,  "population": 10156000, "country": "Thailand"},
    {"name": "Madrid",          "lat": 40.4168,  "lon": -3.7038,   "population": 3280000,  "country": "Spain"},
    {"name": "Barcelona",       "lat": 41.3851,  "lon": 2.1734,    "population": 1620000,  "country": "Spain"},
    {"name": "Rome",            "lat": 41.9028,  "lon": 12.4964,   "population": 2761477,  "country": "Italy"},
    {"name": "Amsterdam",       "lat": 52.3676,  "lon": 4.9041,    "population": 873000,   "country": "Netherlands"},
]

def search_cities(query):
    """Search for cities matching the query, sorted by population (descending)."""
    query_lower = query.lower()
    matching_cities = [
        city for city in MAJOR_CITIES 
        if query_lower in city["name"].lower()
    ]
    # Sort by population descending
    matching_cities.sort(key=lambda x: x["population"], reverse=True)
    return matching_cities

def get_city_display_text(city):
    """Format city display text with population."""
    return f"{city['name']} ({city['population']:,} people)"

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

def get_local_hour(timezone_offset):
    """Get the current hour in the specified timezone offset (seconds from UTC).
    
    Args:
        timezone_offset: Timezone offset in seconds from UTC (from weather API)
    
    Returns:
        Current hour in the city's timezone
    """
    try:
        # timezone_offset is in seconds
        utc_now = datetime.utcnow()
        city_time = utc_now.timestamp() + timezone_offset
        return datetime.fromtimestamp(city_time).hour
    except Exception as e:
        pass
    return datetime.now().hour


# ============== PAGE: HOME ==============
def page_home():
    st.title("Solar Radiation Prediction App")
    
    st.markdown("---")
    st.header("Welcome to the Solar Radiation Prediction Application")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ### Project Overview
        
        The **Solar Radiation Prediction App** is designed to help you estimate solar radiation levels 
        and calculate the solar panel requirements needed to meet your energy needs. This tool combines 
        weather data and machine learning predictions to provide accurate solar energy assessments.
        
        ### Objectives
        
        1. **Predict Solar Radiation**: Use a KNN machine learning model to predict Global Horizontal 
           Irradiance (GHI) based on weather parameters
        2. **Calculate Panel Requirements**: Determine how many solar panels you need to meet your 
           power requirements
        3. **Estimate Energy Generation**: Project annual energy production and potential cost savings
        4. **Location-Based Analysis**: Get weather data automatically for any city worldwide
        
        ### How to Use the Calculator
        
        **Step 1: Navigate to the Prediction Page**
        - Go to the "Prediction" page using the sidebar menu
        
        **Step 2: Enter Location Information**
        - Enter a city name (e.g., "Menlo Park") to automatically fetch current weather data
        - The app uses this data to pre-fill weather parameters
        
        **Step 3: Predict Solar Radiation**
        - Review and adjust weather parameters (temperature, humidity, pressure, wind speed, etc.)
        - Click "Predict GHI" to calculate the GHI (Global Horizontal Irradiance) using our ML model
        
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
        ### Key Metrics
        
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
    st.subheader("What is Solar Radiation?")
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
    st.subheader("About Solar Panel Calculations")
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
    st.subheader("Common Peak Sun Hours by Location")
    peak_hours_data = {
        "Location": ["Northern California (Menlo Park)", "Southern California (Los Angeles)", 
                     "Arizona (Phoenix)", "Texas (Houston)", "Florida (Miami)", "New York (NYC)"],
        "Avg Peak Hours/Day": [5.0, 5.5, 6.0, 5.0, 4.5, 4.0]
    }
    st.dataframe(pd.DataFrame(peak_hours_data), use_container_width=True)


# ============== PAGE: PREDICTION ==============
def page_prediction():
    st.markdown("# ☀️ Solar Radiation Prediction")

    if "selected_city" not in st.session_state:
        st.session_state.selected_city = {"name": "Menlo Park", "lat": 37.4530, "lon": -122.1817}
    if "predicted_ghi" not in st.session_state:
        st.session_state.predicted_ghi = None

    # ── Build deduplicated city list (needed in both columns) ──────────────
    seen_names: set = set()
    unique_cities = []
    for city in sorted(MAJOR_CITIES, key=lambda x: x["population"], reverse=True):
        if city["name"] not in seen_names:
            seen_names.add(city["name"])
            unique_cities.append(city)
    city_options = [get_city_display_text(city) for city in unique_cities]

    # On first ever load city_selectbox has no value, so Streamlit would
    # default to index 0 (Tokyo — highest population) and overwrite the
    # Menlo Park default. Seed it to match selected_city instead.
    if "city_selectbox" not in st.session_state:
        default_name = st.session_state.selected_city.get("name", "")
        default_label = next(
            (get_city_display_text(c) for c in unique_cities if c["name"] == default_name),
            city_options[0],
        )
        st.session_state["city_selectbox"] = default_label

    # Pending-city written by a button click must be applied before the
    # selectbox widget is instantiated (cannot modify key after creation).
    if "_pending_city" in st.session_state:
        pending = st.session_state.pop("_pending_city")
        st.session_state["city_selectbox"] = get_city_display_text(pending)

    # ── Two-column layout: left = city, right = weather inputs ────────────
    left, right = st.columns([1, 2])


    # ── LEFT PANEL: city selection ─────────────────────────────────────────
    with left:
        selected_option = st.selectbox(
            "📍 Search city:",
            options=city_options,
            key="city_selectbox",
            help="Type any part of a city name to filter the list",
        )
        selected_idx = city_options.index(selected_option)

        st.session_state.selected_city = unique_cities[selected_idx]

        # 6 popular cities in 3 columns = 2 compact rows
        st.caption("Popular cities:")
        popular = unique_cities[:6]
        selected_name = st.session_state.selected_city.get("name", "")
        btn_cols = st.columns(3)
        for i, city in enumerate(popular):
            is_selected = city["name"] == selected_name
            label = f"✅ {city['name']}" if is_selected else city["name"]
            if btn_cols[i % 3].button(label, key=f"btn_{city['name']}", use_container_width=True):
                st.session_state["_pending_city"] = city
                st.rerun()

        # Single-line selected location summary
        city = st.session_state.selected_city["name"]
        lat  = st.session_state.selected_city["lat"]
        lon  = st.session_state.selected_city["lon"]
        st.markdown(f"#### 📌 {city}")
        st.caption(f"{lat:.3f}°N, {lon:.3f}°E")


    # ── RIGHT PANEL: weather inputs + prediction ───────────────────────────
    with right:
        st.subheader("🌤️ Weather Features")

        # Fetch live weather for the selected city
        weather_data = fetch_weather_data(lat, lon, city)

        default_humidity    = 46
        default_pressure    = 986
        default_temperature = 29.6
        default_wind_speed  = 2.9

        if weather_data:
            default_humidity    = weather_data.get('main', {}).get('humidity', default_humidity)
            default_pressure    = weather_data.get('main', {}).get('pressure', default_pressure)
            kelvin_temp         = weather_data.get('main', {}).get('temp', 302.15)
            default_temperature = kelvin_temp - 273.15
            default_wind_speed  = weather_data.get('wind', {}).get('speed', default_wind_speed)
            st.caption(f"Live weather loaded for {city}")

        default_dew_point = default_temperature - ((100 - default_humidity) / 5.)
        timezone_offset   = weather_data.get('timezone', 0) if weather_data else 0
        default_hour      = get_local_hour(timezone_offset)

        # Flush fresh defaults into session state when the city changes so
        # number_input widgets (which ignore value= after first render) update.
        city_key = f"{lat:.4f}_{lon:.4f}"
        if st.session_state.get("_last_city_key") != city_key:
            st.session_state["_last_city_key"] = city_key
            st.session_state["hour"]     = int(default_hour)
            st.session_state["temp"]     = float(round(default_temperature, 2))
            st.session_state["dew"]      = float(round(default_dew_point, 2))
            st.session_state["humidity"] = int(default_humidity)
            st.session_state["pressure"] = int(default_pressure)
            st.session_state["wind"]     = float(round(default_wind_speed, 2))
            st.session_state["albedo"]   = 0.15

        # Guard: ensure float fields stay float and int fields stay int so
        # Streamlit never sees a type mismatch between value and min/max.
        for _key in ("temp", "dew", "wind", "albedo"):
            if _key in st.session_state and isinstance(st.session_state[_key], int):
                st.session_state[_key] = float(st.session_state[_key])
        for _key in ("hour", "humidity", "pressure"):
            if _key in st.session_state and isinstance(st.session_state[_key], float):
                st.session_state[_key] = int(st.session_state[_key])

        w1, w2 = st.columns(2)
        with w1:
            hour              = st.number_input("Hour (0–23)",          min_value=0,     max_value=23,   value=int(default_hour),             key="hour")
            temperature       = st.number_input("Temperature (°C)",     min_value=-50.0, max_value=60.0, value=float(default_temperature),    key="temp")
            dew_point         = st.number_input("Dew Point (°C)",       min_value=-50.0, max_value=40.0, value=float(default_dew_point),      key="dew")
            relative_humidity = st.number_input("Relative Humidity (%)", min_value=0,    max_value=100,  value=int(default_humidity),         key="humidity")
        with w2:
            surface_albedo = st.number_input("Surface Albedo",      min_value=0.0, max_value=1.0,    value=0.15,                        key="albedo")
            pressure       = st.number_input("Pressure (hPa)",      min_value=800, max_value=1100,   value=int(default_pressure),       key="pressure")
            wind_speed     = st.number_input("Wind Speed (m/s)",    min_value=0.0, max_value=50.0,   value=float(default_wind_speed),   key="wind")

        feature_names = ['Hour', 'Temperature', 'Dew Point', 'Relative Humidity', 'Surface Albedo', 'Pressure', 'Wind Speed']
        df = pd.DataFrame({
            "Hour": hour, "Temperature": temperature, "Dew Point": dew_point,
            "Relative Humidity": relative_humidity, "Surface Albedo": surface_albedo,
            "Pressure": pressure, "Wind Speed": wind_speed,
        }, index=[0])

        st.divider()
        if st.button("⚡ Predict GHI", key="predict_btn", use_container_width=True):
            if knn is None:
                st.error("Model not loaded. Ensure `knn.pkl` exists.")
            else:
                try:
                    prediction = knn.predict(df[feature_names].values)
                    st.session_state.predicted_ghi = float(prediction[0])
                except Exception as e:
                    st.error(f"Prediction failed: {e}")

        if st.session_state.predicted_ghi is not None:
            st.success(f"### {st.session_state.predicted_ghi:.2f} W/m²  — Predicted GHI")
            st.caption("Head to the Calculator page to size your solar panel system.")
        else:
            st.info("Click **Predict GHI** to generate a prediction.")



# ============== PAGE: CALCULATOR ==============
def page_calculator():
    st.markdown("# 🔆 Solar Panel Calculator")

    if "predicted_ghi" not in st.session_state or st.session_state.predicted_ghi is None:
        st.warning("No GHI prediction yet — go to the Prediction page and click **Predict GHI** first.")
        return

    # Persist calculation results across reruns
    if "calc_results" not in st.session_state:
        st.session_state.calc_results = None

    brands = [
        {"Brand": "SunPower",      "Efficiency (%)": 22.8, "Area (m²)": 1.63, "Nominal Power (W)": 430},
        {"Brand": "LG",            "Efficiency (%)": 20.4, "Area (m²)": 1.70, "Nominal Power (W)": 400},
        {"Brand": "Panasonic",     "Efficiency (%)": 20.3, "Area (m²)": 1.70, "Nominal Power (W)": 405},
        {"Brand": "Jinko (Generic)","Efficiency (%)": 18.4, "Area (m²)": 1.70, "Nominal Power (W)": 375},
        {"Brand": "Custom",        "Efficiency (%)": None,  "Area (m²)": None,  "Nominal Power (W)": None},
    ]
    brands_df = pd.DataFrame(brands)

    left, right = st.columns([1, 1])

    # ── LEFT: all inputs ──────────────────────────────────────────────────
    with left:
        _city_name = st.session_state.get("selected_city", {}).get("name", "Unknown")
        st.caption(f"📍 {_city_name} · GHI: **{st.session_state.predicted_ghi:.1f} W/m²**")

        required_power = st.number_input("Required Power Output (W)", min_value=0.0,
                                         max_value=1_000_000.0, value=800.0, key="required_power")

        chosen_brand = st.selectbox("Panel brand", [b["Brand"] for b in brands], key="brand_select")

        if chosen_brand == "Custom":
            c1, c2, c3 = st.columns(3)
            panel_efficiency = c1.number_input("Efficiency (%)", 0.0, 100.0, 18.0, key="custom_eff")
            panel_area       = c2.number_input("Area (m²)",      0.1,  10.0,  1.7, key="custom_area")
            nominal_power    = c3.number_input("Nominal W",       1.0, 2000.0, 400.0, key="custom_power")
        else:
            row = brands_df[brands_df["Brand"] == chosen_brand].iloc[0]
            panel_efficiency = float(row["Efficiency (%)"])
            panel_area       = float(row["Area (m²)"])
            nominal_power    = float(row["Nominal Power (W)"])
            st.caption(f"Efficiency {panel_efficiency}% · Area {panel_area} m² · {nominal_power:.0f} W nominal")

        derate = st.slider("System Derate Factor", 0.5, 1.0, 0.77, 0.01, key="derate",
                           help="Accounts for inverter, wiring, temperature losses. Typical: 0.77",
                           disabled=True)

        peak_sun_hours = st.slider("Peak Sun Hours / Day", 2.0, 8.0, 5.0, 0.5, key="peak_hours")

        # Auto-detect electricity rate from selected city's country.
        # Look up country from MAJOR_CITIES directly (more reliable than
        # trusting the stored selected_city dict which may lack the field).
        selected_city = st.session_state.get("selected_city", {})
        city_name = selected_city.get("name", "")
        country = next(
            (c["country"] for c in MAJOR_CITIES if c["name"] == city_name),
            selected_city.get("country", ""),
        )
        auto_rate = ELECTRICITY_RATES.get(country, DEFAULT_RATE)
        # Sync widget when city changes (track by city name, not country,
        # so switching between two cities in the same country still updates).
        if st.session_state.get("_last_rate_city") != city_name:
            st.session_state["_last_rate_city"] = city_name
            st.session_state["electricity_rate"] = float(auto_rate)
        rate_label = f"Electricity rate ($/kWh) — {country} avg" if country else "Electricity rate ($/kWh)"
        electricity_rate = st.number_input(rate_label, 0.0, value=float(auto_rate), key="electricity_rate")

        system_cost = st.number_input("Installation cost ($)", 0.0, value=8000.0, key="system_cost")

        override_ghi = st.checkbox("Override GHI", key="override_ghi")
        ghi_value = (st.number_input("Override GHI (W/m²)", 0.0, value=800.0, key="override_ghi_val")
                     if override_ghi else st.session_state.predicted_ghi)

        if st.button("⚡ Calculate Panels", key="calculate_btn", use_container_width=True):
            if not ghi_value or ghi_value <= 0:
                st.error("GHI must be > 0.")
            else:
                try:
                    ppp = ghi_value * panel_area * (panel_efficiency / 100.0) * derate
                    if ppp <= 0:
                        st.error("Power per panel ≤ 0 — check inputs.")
                    else:
                        n          = ceil(required_power / ppp)
                        area_total = n * panel_area
                        sys_power  = n * ppp
                        e_per_year = (ppp / 1000) * peak_sun_hours * 365
                        e_total    = e_per_year * n
                        savings    = e_total * electricity_rate
                        payback    = (system_cost / savings) if savings > 0 else None
                        st.session_state.calc_results = {
                            "ghi": ghi_value, "ppp": ppp, "n": n,
                            "area_total": area_total, "sys_power": sys_power,
                            "e_total": e_total, "e_per_year": e_per_year,
                            "savings": savings, "payback": payback,
                            "electricity_rate": electricity_rate,
                        }
                except Exception as e:
                    st.error(f"Calculation failed: {e}")

    # ── RIGHT: results ────────────────────────────────────────────────────
    with right:
        r = st.session_state.calc_results
        if r is None:
            st.info("Configure inputs on the left and click **Calculate Panels**.")
        else:
            st.markdown("#### Results")
            # Row 1 — panel sizing
            m1, m2 = st.columns(2)
            m1.metric("Panels",       r["n"])
            m2.metric("Array Area",   f"{r['area_total']:.1f} m²")
            # Row 2 — power & energy
            m3, m4 = st.columns(2)
            m3.metric("System Power",  f"{r['sys_power']:.0f} W")
            m4.metric("Annual Energy", f"{r['e_total']:,.0f} kWh")
            # Row 3 — finance
            m5, m6 = st.columns(2)
            m5.metric("Annual Savings", f"${r['savings']:,.0f}")
            m6.metric("Payback",        f"{r['payback']:.1f} yr" if r["payback"] else "—")
            # Detail caption
            st.caption(
                f"Per-panel: {r['ppp']:.1f} W · {r['e_per_year']:,.0f} kWh/yr  |  "
                f"GHI: {r['ghi']:.0f} W/m²  |  Rate: ${r['electricity_rate']}/kWh"
            )


# ============== PAGE: ABOUT ==============
def page_about():
    st.title("About This Application")
    
    st.markdown("---")
    st.header("Application Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Technology Stack")
        st.markdown("""
        - **Frontend**: Streamlit (Python web framework)
        - **ML Model**: K-Nearest Neighbors (KNN)
        - **Weather Data**: OpenWeatherMap API
        - **Data Processing**: Pandas, NumPy
        - **Visualization**: Streamlit charts and metrics
        """)
        
        st.subheader("Model Details")
        st.markdown("""
        - **Algorithm**: K-Nearest Neighbors (KNN)
        - **Input Features**: 7 weather parameters
        - **Target Variable**: Global Horizontal Irradiance (GHI)
        - **Pre-training**: Normalized input features
        - **Output**: Solar radiation prediction in W/m²
        """)
    
    with col2:
        st.subheader("Data Sources")
        st.markdown("""
        - **Weather Data**: OpenWeatherMap API
          - Current conditions for any location
          - Temperature, humidity, pressure, wind speed
        
        - **Solar Data**: Training model based on 
          real solar measurement datasets
        
        - **Location Data**: Geographic coordinate lookup
        """)
        
        st.subheader("Features")
        st.markdown("""
        - Real-time weather integration
        - ML-based solar radiation prediction
        - Customizable panel specifications
        - Annual energy generation estimates
        - Financial savings calculations
        - Location-based analysis
        """)
    
    st.markdown("---")
    st.subheader("How Solar Radiation Prediction Works")
    
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
    st.subheader("Solar Panel Efficiency Factors")
    
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
    st.subheader("FAQ")
    
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
    st.subheader("Support & Feedback")
    st.markdown("""
    For questions, issues, or suggestions about this application, please refer to the project documentation
    or contact the development team.
    
    **Version**: 2.0 (Multi-page interface)  
    **Last Updated**: January 2026
    """)


# ============== MAIN APP NAVIGATION ==============
def main():
    st.sidebar.title("Navigation")
    
    page_options = {
        "Home": page_home,
        "Prediction": page_prediction,
        "Calculator": page_calculator,
        "About": page_about,
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
