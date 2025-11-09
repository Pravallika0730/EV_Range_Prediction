import streamlit as st
import numpy as np
import pandas as pd
import joblib
from catboost import CatBoostRegressor
from geopy.distance import geodesic
import folium
import requests
import os
from dotenv import load_dotenv
from streamlit.components.v1 import html  # for direct HTML render

st.set_page_config(layout="wide")

# --- ENV / API KEY ---
load_dotenv()
api_key = os.getenv("ORS_API_KEY")
if not api_key:
    st.error("OpenRouteService API key not found. Set ORS_API_KEY in a .env file.")
    st.stop()

# ---- cache model/scaler loads
@st.cache_resource
def load_models():
    cat = CatBoostRegressor()
    cat.load_model("catboost_ev_model.cbm")
    rf = joblib.load("random_forest_model.pkl")
    sc = joblib.load("scaler.pkl")
    return cat, rf, sc

cat_model, rf_model, scaler = load_models()

traffic_mapping = {"Low": 1, "Medium": 2, "High": 3}
battery_health_mapping = {"Healthy": 0, "Aging": 1, "Degraded": 2}

st.title("EV Range Prediction & Optimized Bike Route Finder")
st.write("Enter the details below to predict the estimated EV range and find the best bike-friendly route to a charging station.")

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

# optional sanity hint
if battery_end > battery_start:
    st.info("Tip: Battery end > start — double-check your inputs.")

battery_health_encoded = battery_health_mapping.get(battery_health, 0)
traffic_encoded = traffic_mapping.get(traffic_level, 1)

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

input_data = input_data.apply(pd.to_numeric, errors='coerce')
if input_data.isna().sum().sum() > 0:
    st.warning("⚠️ Some input values were NaN and have been replaced with 0.")
    input_data = input_data.fillna(0)

input_scaled = scaler.transform(input_data)

if 'catboost_pred' not in st.session_state:
    st.session_state.catboost_pred = None
if 'rf_pred' not in st.session_state:
    st.session_state.rf_pred = None

if st.button("Predict EV Range 🚀"):
    try:
        st.session_state.rf_pred = rf_model.predict(input_data)[0]
        st.session_state.catboost_pred = cat_model.predict(input_scaled)[0]
        st.subheader("Predicted EV Range:")
        st.write(f"🚗 **CatBoost Prediction:** {st.session_state.catboost_pred:.2f} km")
        st.write(f"🌲 **Random Forest Prediction:** {st.session_state.rf_pred:.2f} km")
    except Exception as e:
        st.error(f"⚠️ Prediction Error: {e}")

if st.session_state.catboost_pred is not None:
    st.subheader("Predicted EV Range (Previous Prediction):")
    st.write(f"🚗 **CatBoost Prediction:** {st.session_state.catboost_pred:.2f} km")
    st.write(f"🌲 **Random Forest Prediction:** {st.session_state.rf_pred:.2f} km")

# --- routing ---
st.subheader("Find the Best Bike-Friendly Route to a Charging Station")
start_lat = st.number_input("Current Latitude", value=12.9716)
start_lon = st.number_input("Current Longitude", value=77.5946)
station_lat = st.number_input("Charging Station Latitude", value=12.9352)
station_lon = st.number_input("Charging Station Longitude", value=77.6245)

if st.session_state.catboost_pred is not None:
    # Base map
    map_center = [(start_lat + station_lat) / 2, (start_lon + station_lon) / 2]
    route_map = folium.Map(location=map_center, zoom_start=13)
    folium.Marker([start_lat, start_lon], tooltip="Current Location", icon=folium.Icon(color='blue')).add_to(route_map)
    folium.Marker([station_lat, station_lon], tooltip="Charging Station", icon=folium.Icon(color='green')).add_to(route_map)

    # Helper: draw route from ORS GeoJSON FeatureCollection
    def draw_route_from_geojson(fc_json):
        if not isinstance(fc_json, dict):
            return False
        feats = fc_json.get("features", [])
        if not feats:
            return False

        feat = feats[0]
        geom = feat.get("geometry", {})
        coords = geom.get("coordinates", [])
        gtype = geom.get("type", "")

        def to_latlon(seq):
            return [(pt[1], pt[0]) for pt in seq]  # ORS is [lon, lat] -> (lat, lon)

        route_latlon = []
        if gtype == "LineString":
            route_latlon = to_latlon(coords)
        elif gtype == "MultiLineString":
            for seg in coords:
                route_latlon.extend(to_latlon(seg))
        else:
            st.warning(f"Unexpected geometry type from ORS: {gtype}")

        if route_latlon:
            folium.PolyLine(route_latlon, weight=5, opacity=0.9).add_to(route_map)
            route_map.fit_bounds(route_latlon)
            summary = feat.get("properties", {}).get("summary", {})
            dist_km = summary.get("distance", 0) / 1000
            dur_min = summary.get("duration", 0) / 60
            if dist_km or dur_min:
                st.caption(f"✅ ORS route • ~{dist_km:.2f} km • ~{dur_min:.0f} min")
            else:
                st.caption("✅ ORS route loaded")
            return True
        return False

    # ORS POST
    got_route = False
    url_post = "https://api.openrouteservice.org/v2/directions/cycling-regular"
    headers = {'Authorization': api_key, 'Content-Type': 'application/json'}
    body = {'coordinates': [[start_lon, start_lat], [station_lon, station_lat]], 'format': 'geojson'}

    try:
        resp = requests.post(url_post, headers=headers, json=body, timeout=20)
        if resp.status_code == 200:
            data_json = resp.json()
            got_route = draw_route_from_geojson(data_json)
            if not got_route:
                st.warning("POST returned 200 but no features; trying GET fallback…")
                st.caption(f"POST response head: {resp.text[:200]}")
        else:
            st.error(f"ORS POST {resp.status_code}: {resp.text[:300]}")
    except requests.RequestException as e:
        st.error(f"ORS POST failed: {e}")

    # ORS GET fallback
    if not got_route:
        try:
            url_get = (
                "https://api.openrouteservice.org/v2/directions/cycling-regular"
                f"?api_key={api_key}&start={start_lon},{start_lat}&end={station_lon},{station_lat}"
            )
            resp2 = requests.get(url_get, timeout=20)
            if resp2.status_code == 200:
                data_json2 = resp2.json()
                got_route = draw_route_from_geojson(data_json2)
                if not got_route:
                    st.error("GET returned 200 but no features.")
                    st.caption(f"GET response head: {resp2.text[:200]}")
            else:
                st.error(f"ORS GET {resp2.status_code}: {resp2.text[:300]}")
        except requests.RequestException as e:
            st.error(f"ORS GET failed: {e}")

    # Fallback straight line
    if not got_route:
        folium.PolyLine([(start_lat, start_lon), (station_lat, station_lon)], weight=3).add_to(route_map)
        st.caption("⚠️ Showing straight line fallback")

    # Render Folium map via HTML (robust)
    map_html = route_map.get_root().render()
    html(map_html, height=520)

    # Feasibility vs predicted range
    trip_km = geodesic((start_lat, start_lon), (station_lat, station_lon)).km
    st.write(f"Distance to Charging Station: **{trip_km:.2f} km**")
    if trip_km <= st.session_state.catboost_pred:
        st.success("Charging station is within the predicted range!")
    else:
        st.warning("Charging station is outside the predicted range.")
else:
    st.info("Please run the prediction first to verify the feasibility of the trip.")
