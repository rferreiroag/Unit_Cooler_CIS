"""
Streamlit Dashboard for HVAC Unit Cooler Digital Twin

Real-time monitoring and prediction interface for naval HVAC systems.
"""

import streamlit as st
import requests
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import time
import json
from pathlib import Path

# Page configuration
st.set_page_config(
    page_title="HVAC Digital Twin Dashboard",
    page_icon="🌡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# API configuration
API_URL = st.secrets.get("API_URL", "http://localhost:8000")

# Custom CSS
st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .stAlert {
        margin-top: 10px;
    }
</style>
""", unsafe_allow_html=True)


def check_api_health():
    """Check if API is available."""
    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        if response.status_code == 200:
            return True, response.json()
        return False, None
    except Exception as e:
        return False, str(e)


def make_prediction(input_data):
    """Make prediction using the API."""
    try:
        response = requests.post(
            f"{API_URL}/predict",
            json=input_data,
            timeout=10
        )
        if response.status_code == 200:
            return True, response.json()
        return False, f"Error: {response.status_code} - {response.text}"
    except Exception as e:
        return False, str(e)


def create_gauge_chart(value, title, min_val, max_val, optimal_range=None):
    """Create a gauge chart for displaying metrics."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        title={'text': title},
        gauge={
            'axis': {'range': [min_val, max_val]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [min_val, optimal_range[0] if optimal_range else min_val], 'color': "lightgray"},
                {'range': [optimal_range[0] if optimal_range else min_val,
                          optimal_range[1] if optimal_range else max_val], 'color': "lightgreen"},
                {'range': [optimal_range[1] if optimal_range else max_val, max_val], 'color': "lightgray"}
            ] if optimal_range else [],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': max_val * 0.9
            }
        }
    ))
    fig.update_layout(height=250, margin=dict(l=20, r=20, t=50, b=20))
    return fig


def main():
    """Main dashboard application."""

    # Header
    st.title("🌡️ HVAC Unit Cooler Digital Twin")
    st.markdown("### Real-time Monitoring and Prediction System")

    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")

        # API Status
        api_healthy, health_data = check_api_health()
        if api_healthy:
            st.success("✅ API Connected")
            with st.expander("API Details"):
                st.json(health_data)
        else:
            st.error(f"❌ API Disconnected")
            st.info(f"API URL: {API_URL}")
            st.warning("Make sure the API is running:\n```uvicorn api.main:app --reload```")

        st.markdown("---")

        # Mode selection
        mode = st.radio(
            "Operation Mode",
            ["Manual Input", "Real-time Monitoring", "Historical Analysis"]
        )

        st.markdown("---")

        # About
        with st.expander("ℹ️ About"):
            st.markdown("""
            **HVAC Digital Twin v1.0**

            - **Model**: LightGBM (R²=0.993-1.0)
            - **Latency**: <0.022 ms (P95)
            - **Targets**: UCAOT, UCWOT, UCAF
            - **Deployment**: ONNX Runtime

            Sprint 7: Real-time Integration
            """)

    # Main content based on mode
    if mode == "Manual Input":
        show_manual_input()
    elif mode == "Real-time Monitoring":
        show_realtime_monitoring()
    else:
        show_historical_analysis()


def show_manual_input():
    """Manual input mode for single predictions."""
    st.header("📝 Manual Input Mode")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Input Parameters")

        # Water circuit inputs
        st.markdown("**Water Circuit**")
        ucwit = st.number_input(
            "Water Inlet Temperature (°C)",
            min_value=0.0, max_value=50.0, value=7.5, step=0.1,
            help="UCWIT: Temperatura de entrada del agua"
        )
        ucwf = st.number_input(
            "Water Flow Rate (L/min)",
            min_value=0.0, max_value=100.0, value=15.0, step=0.5,
            help="UCWF: Caudal de agua"
        )

        # Air circuit inputs
        st.markdown("**Air Circuit**")
        ucait = st.number_input(
            "Air Inlet Temperature (°C)",
            min_value=0.0, max_value=50.0, value=25.0, step=0.1,
            help="UCAIT: Temperatura de entrada del aire"
        )
        ucaih = st.number_input(
            "Air Inlet Humidity (%)",
            min_value=0.0, max_value=100.0, value=50.0, step=1.0,
            help="UCAIH: Humedad del aire de entrada"
        )

        # Control parameters
        st.markdown("**Control & Environment**")
        uctsp = st.number_input(
            "Temperature Setpoint (°C)",
            min_value=15.0, max_value=35.0, value=21.0, step=0.5,
            help="UCTSP: Temperatura objetivo"
        )
        ambt = st.number_input(
            "Ambient Temperature (°C)",
            min_value=0.0, max_value=50.0, value=22.0, step=0.1,
            help="AMBT: Temperatura ambiente"
        )

        # Optional parameters
        with st.expander("Optional Parameters"):
            ucfms = st.number_input("Fan Motor Speed", value=1500.0)
            ucfmv = st.number_input("Fan Motor Voltage", value=220.0)
            cppr = st.number_input("Chiller Primary Pump Rate", value=80.0)
            cpdp = st.number_input("Chiller Primary Differential Pressure", value=150.0)

    with col2:
        st.subheader("Prediction Results")

        if st.button("🚀 Make Prediction", type="primary", use_container_width=True):
            # Prepare input data
            input_data = {
                "UCWIT": ucwit,
                "UCAIT": ucait,
                "UCWF": ucwf,
                "UCAIH": ucaih,
                "AMBT": ambt,
                "UCTSP": uctsp,
                "UCFMS": ucfms,
                "UCFMV": ucfmv,
                "CPPR": cppr,
                "CPDP": cpdp
            }

            with st.spinner("Making prediction..."):
                success, result = make_prediction(input_data)

            if success:
                st.success("✅ Prediction successful!")

                # Display metrics
                metrics_col1, metrics_col2, metrics_col3 = st.columns(3)

                with metrics_col1:
                    st.metric(
                        "Air Outlet Temp (UCAOT)",
                        f"{result['UCAOT']:.2f} °C",
                        delta=f"{result['UCAOT'] - ucait:.2f} °C"
                    )

                with metrics_col2:
                    st.metric(
                        "Water Outlet Temp (UCWOT)",
                        f"{result['UCWOT']:.2f} °C",
                        delta=f"{result['UCWOT'] - ucwit:.2f} °C"
                    )

                with metrics_col3:
                    st.metric(
                        "Air Flow (UCAF)",
                        f"{result['UCAF']:.0f} m³/h"
                    )

                # Thermal power and inference time
                st.markdown("---")
                perf_col1, perf_col2 = st.columns(2)

                with perf_col1:
                    st.metric(
                        "Thermal Power",
                        f"{result['Q_thermal']:.2f} kW"
                    )

                with perf_col2:
                    st.metric(
                        "Inference Time",
                        f"{result['inference_time_ms']:.3f} ms"
                    )

                # Visualization
                st.markdown("---")
                st.subheader("Temperature Flow Diagram")

                fig = go.Figure()

                # Water circuit
                fig.add_trace(go.Scatter(
                    x=[0, 1],
                    y=[ucwit, result['UCWOT']],
                    mode='lines+markers+text',
                    name='Water Circuit',
                    line=dict(color='blue', width=3),
                    marker=dict(size=12),
                    text=[f"Inlet: {ucwit:.1f}°C", f"Outlet: {result['UCWOT']:.1f}°C"],
                    textposition='top center'
                ))

                # Air circuit
                fig.add_trace(go.Scatter(
                    x=[0, 1],
                    y=[ucait, result['UCAOT']],
                    mode='lines+markers+text',
                    name='Air Circuit',
                    line=dict(color='red', width=3),
                    marker=dict(size=12),
                    text=[f"Inlet: {ucait:.1f}°C", f"Outlet: {result['UCAOT']:.1f}°C"],
                    textposition='bottom center'
                ))

                fig.update_layout(
                    title="Heat Exchanger Performance",
                    xaxis_title="Position",
                    yaxis_title="Temperature (°C)",
                    xaxis=dict(tickvals=[0, 1], ticktext=["Inlet", "Outlet"]),
                    height=400
                )

                st.plotly_chart(fig, use_container_width=True)

                # Raw response
                with st.expander("Raw API Response"):
                    st.json(result)
            else:
                st.error(f"❌ Prediction failed: {result}")


def show_realtime_monitoring():
    """Real-time monitoring mode."""
    st.header("📊 Real-time Monitoring")

    st.info("⚠️ This feature requires active data streaming. Implementation pending.")

    # Placeholder for real-time charts
    chart_placeholder = st.empty()
    metrics_placeholder = st.empty()

    # Simulation (to be replaced with actual streaming)
    if st.button("Start Simulation"):
        for i in range(10):
            # Generate simulated data
            current_time = datetime.now()

            with metrics_placeholder.container():
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("UCAOT", f"{20 + np.random.randn():.2f} °C")
                col2.metric("UCWOT", f"{10 + np.random.randn():.2f} °C")
                col3.metric("UCAF", f"{5000 + np.random.randn()*100:.0f} m³/h")
                col4.metric("Q_thermal", f"{15 + np.random.randn():.2f} kW")

            time.sleep(1)


def show_historical_analysis():
    """Historical analysis mode."""
    st.header("📈 Historical Analysis")

    st.info("⚠️ This feature requires historical data storage. Implementation pending.")

    # Placeholder for historical analysis
    st.markdown("""
    **Features to be implemented:**
    - Load historical predictions from database
    - Trend analysis and visualization
    - Performance metrics over time
    - Drift detection reports
    - Export capabilities
    """)


if __name__ == "__main__":
    main()
