import streamlit as st
import numpy as np
import pandas as pd
import joblib
from catboost import CatBoostRegressor

#Load trained models
cat_model = CatBoostRegressor()
cat_model.load_model("catboost_ev_model.cbm")  # Load CatBoost model
rf_model = joblib.load("random_forest_model.pkl")  # Load Random Forest model

#Load saved scaler (for CatBoost preprocessing)
scaler = joblib.load("scaler.pkl")

#Manually define mapping for categorical values
traffic_mapping = {"Low": 1, "Medium": 2, "High": 3}
battery_health_mapping = {"Healthy": 0, "Aging": 1, "Degraded": 2}

#Streamlit UI
st.title("EV Range Prediction App")
st.write("Enter the details below to predict the estimated EV range.")

#User Inputs
ride_duration = st.number_input("Ride Duration (minutes)", min_value=1, value=30)
distance = st.number_input("Distance Traveled (km)", min_value=0.1, value=10.5)
battery_start = st.number_input("Battery Start Percentage (%)", min_value=0, max_value=100, value=80)
battery_end = st.number_input("Battery End Percentage (%)", min_value=0, max_value=100, value=65)
avg_speed = st.number_input("Average Speed (km/h)", min_value=1, value=40)
elevation = st.number_input("Elevation (%)", min_value=0, value=5)
temperature = st.number_input("Temperature (°C)", min_value=-10, max_value=50, value=28)
traffic_level = st.selectbox("Traffic Level", ["Low", "Medium", "High"])
load_weight = st.number_input("Load Weight (kg)", min_value=0, value=75)
energy_consumed = st.number_input("Energy Consumed (Wh)", min_value=1, value=500)
battery_health = st.selectbox("Battery Health Status", ["Healthy", "Aging", "Degraded"])
efficiency = st.number_input("Efficiency (Wh/km)", min_value=1, value=48)
speed_variation = st.number_input("Speed Variation", min_value=0, value=2)
temp_change = st.number_input("Temperature Change", min_value=-10.0, value=0.5)
battery_usage = st.number_input("Battery Usage (%)", min_value=0, max_value=100, value=15)

#Convert categorical inputs using manual mapping
battery_health_encoded = battery_health_mapping.get(battery_health, 0)  # Default: "Healthy" → 0
traffic_encoded = traffic_mapping.get(traffic_level, 1)  # Default: "Low" → 1

#Prepare input data as a DataFrame
input_data = pd.DataFrame([[
    ride_duration, distance, battery_start, battery_end, avg_speed,
    elevation, temperature, traffic_encoded, load_weight,
    energy_consumed, battery_health_encoded, efficiency, speed_variation,
    temp_change, battery_usage
]], columns=[
    "Ride Duration (minutes)", "Distance Traveled (km)", "Battery Start Percentage (%)",
    "Battery End Percentage (%)", "Average Speed (km/h)", "Elevation (%)", "Temperature (°C)",
    "Traffic_Level", "Load Weight (kg)", "Energy Consumed (Wh)", "Battery Health Status",
    "Efficiency (Wh/km)", "Speed_Variation", "Temp_Change", "Battery_Usage"
])

# **Ensure all values are numeric**
input_data = input_data.apply(pd.to_numeric, errors='coerce')

# **Check for NaN values and replace them**
if input_data.isna().sum().sum() > 0:
    st.warning("⚠️ Warning: Some input values were NaN and have been replaced with 0.")
    input_data = input_data.fillna(0)  # Replace NaN values with 0

# Scale input data for CatBoost (Random Forest does not need scaling)
input_scaled = scaler.transform(input_data)

# Prediction Button
if st.button("Predict EV Range 🚀"):
    try:
        # Predict using Random Forest (raw input, NO SCALING)
        rf_pred = rf_model.predict(input_data)[0]

        # Predict using CatBoost (scaled input)
        catboost_pred = cat_model.predict(input_scaled)[0]

        # Display results
        st.subheader("Predicted EV Range:")
        st.write(f"🚗 **CatBoost Prediction:** {catboost_pred:.2f} km")
        st.write(f"🌲 **Random Forest Prediction:** {rf_pred:.2f} km")

    except Exception as e:
        st.error(f"⚠️ Prediction Error: {e}")



# Map Integration
from geopy.distance import geodesic
import folium
from streamlit_folium import st_folium

st.subheader("Find the Best Route to a Charging Station")

# Input current and charging station locations
start_lat = st.number_input("Current Latitude", value=12.9716)
start_lon = st.number_input("Current Longitude", value=77.5946)
station_lat = st.number_input("Charging Station Latitude", value=12.9352)
station_lon = st.number_input("Charging Station Longitude", value=77.6245)

# Calculate distance
distance_to_station = geodesic((start_lat, start_lon), (station_lat, station_lon)).km

# Show distance and feasibility
st.write(f"Distance to Charging Station: **{distance_to_station:.2f} km**")

if distance_to_station <= catboost_pred:
    st.success("Charging station is within the predicted range!")
else:
    st.warning("Charging station is outside the predicted range. Consider an alternate route or charging option.")

# Display the map
map_center = [(start_lat + station_lat) / 2, (start_lon + station_lon) / 2]
route_map = folium.Map(location=map_center, zoom_start=13)

# Add markers
folium.Marker([start_lat, start_lon], tooltip="Current Location", icon=folium.Icon(color='blue')).add_to(route_map)
folium.Marker([station_lat, station_lon], tooltip="Charging Station", icon=folium.Icon(color='green')).add_to(route_map)

# Draw a line between the points
folium.PolyLine([(start_lat, start_lon), (station_lat, station_lon)], color="blue", weight=2.5, opacity=1).add_to(route_map)

# Render map in Streamlit
st_folium(route_map, width=700, height=500)