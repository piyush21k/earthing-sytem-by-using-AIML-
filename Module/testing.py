import streamlit as st
import numpy as np
import pandas as pd
import joblib
import time
import matplotlib.pyplot as plt

st.set_page_config(page_title="Real-Time Earthing Monitor", layout="wide")

# Load model and scaler
try:
    model = joblib.load("earthing_maintenance_model.pkl")
    scaler = joblib.load("scaler.pkl") 

    st.title("📡 Real-Time Earthing System Monitoring Dashboard")
    st.markdown("This dashboard simulates real-time updates of sensor values and predictions.")

    # Placeholder for charts
    chart_placeholder = st.empty()
    prediction_placeholder = st.empty()

    # Initialize data buffer
    max_points = 300 # plot last 300 seconds
    data_buffer = pd.DataFrame(columns=[
        "timestamp", "Soil_Resistivity", "Ground_Resistance",
        "Ambient_Temperature", "Soil_Moisture",
        "Leakage_Current", "Fault_Current_Events", "Corrosion_Level", "Prediction"
    ])

    # Simulate or fetch real-time data in a loop
    for i in range(300):  # Loop for 300 updates (~5 mins)
        current_time = pd.Timestamp.now()

        # Simulated real-time sensor values (replace with live sensor fetch)
        soil_resistivity = np.random.uniform(30, 100)
        ground_resistance = np.random.uniform(1, 7)
        ambient_temp = np.random.uniform(20, 45)
        soil_moisture = np.random.uniform(10, 40)
        leakage_current = np.random.uniform(0.1, 4.5)
        fault_events = np.random.randint(0, 10)
        corrosion_level = np.random.uniform(0.1, 0.9)

        X_input = np.array([[soil_resistivity, ground_resistance, ambient_temp,
                             soil_moisture, leakage_current, fault_events, corrosion_level]])
        X_scaled = scaler.transform(X_input)
        prediction = model.predict(X_scaled)[0]

        # Append to buffer
        new_data = pd.DataFrame([{
            "timestamp": current_time,
            "Soil_Resistivity": soil_resistivity,
            "Ground_Resistance": ground_resistance,
            "Ambient_Temperature": ambient_temp,
            "Soil_Moisture": soil_moisture,
            "Leakage_Current": leakage_current,
            "Fault_Current_Events": fault_events,
            "Corrosion_Level": corrosion_level,
            "Prediction": prediction
        }])
        data_buffer = pd.concat([data_buffer, new_data], ignore_index=True)
        data_buffer = data_buffer.tail(max_points)  # Keep recent N points

        # Plot live chart
        with chart_placeholder.container():
            st.subheader("📉 Live Sensor Readings")
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(data_buffer["timestamp"], data_buffer["Soil_Resistivity"], label="Soil Resistivity")
            ax.plot(data_buffer["timestamp"], data_buffer["Ground_Resistance"], label="Ground Resistance")
            ax.plot(data_buffer["timestamp"], data_buffer["Leakage_Current"], label="Leakage Current")
            ax.set_xlabel("Time")
            ax.set_ylabel("Value")
            ax.legend()
            ax.grid(True)
            st.pyplot(fig)

        # Display latest prediction
        maintenance_text = "🛠️ Maintenance Required" if prediction == 1 else "✅ No Maintenance Needed"
        with prediction_placeholder.container():
            st.subheader("🧠 AI Prediction (Latest)")
            st.success(f"Prediction at {current_time.strftime('%H:%M:%S')} → {maintenance_text}")

        time.sleep(10)  # Wait for 10 seconds before next reading
except Exception as e:
    st.error("🚨 Model or scaler file not found.")
    st.text(f"Error: {e}")
