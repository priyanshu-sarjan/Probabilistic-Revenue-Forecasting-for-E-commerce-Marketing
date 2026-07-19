import streamlit as st
import pandas as pd
import subprocess
import os

st.set_page_config(page_title="AyurTrace Analytics", layout="wide")

st.title("🌿 Ayurvedic E-Commerce SCM & Marketing Forecaster")
st.subheader("Probabilistic Revenue & ROAS Tool with Supply Traceability Integration")

# Sidebar Configuration for Budget & Logistics Simulation[cite: 1]
st.sidebar.header("🎛️ Simulation Controls")
google_spend = st.sidebar.slider("Google Ads Future Budget ($)", 100, 10000, 2000)
meta_spend = st.sidebar.slider("Meta Ads Future Budget ($)", 100, 10000, 2500)
ms_spend = st.sidebar.slider("Microsoft Ads Future Budget ($)", 50, 5000, 500)

st.sidebar.markdown("---")
st.sidebar.subheader("🚚 Supply Chain Risk Gauges")
delay_days = st.sidebar.slider("Predicted Transit/Warehouse Delay (Days)", 0, 15, 3)
purity_score = st.sidebar.slider("Batch Purity Index (%)", 85.0, 100.0, 97.5)

if st.sidebar.button("⚡ Run Forecasting Pipeline"):
    # Dynamically rewrite the csv based on UI simulation inputs
    simed_data = pd.DataFrame({
        'date': pd.date_range(start='2026-07-19', periods=10),
        'google_ads_spend': [google_spend]*10,
        'meta_ads_spend': [meta_spend]*10,
        'ms_ads_spend': [ms_spend]*10,
        'revenue': [8500]*10,
        'roas': [3.4]*10,
        'harvest_seasonality_index': [1.1]*10,
        'warehouse_delay_days': [delay_days]*10,
        'batch_purity_score': [purity_score]*10
    })
    
    os.makedirs('data', exist_ok=True)
    simed_data.to_csv('data/marketing_supply_data.csv', index=False)
    
    # Run the shell script pipeline
    with st.spinner("Processing automated features and running probabilistic engine..."):
        subprocess.run(["bash", "run.sh", "./data", "./pickle/model.pkl", "./output/predictions.csv"], check=True)
    
    # Read generated outputs[cite: 2]
    if os.path.exists("./output/predictions.csv"):
        res_df = pd.read_csv("./output/predictions.csv")
        
        st.success("🎉 Forecast Pipeline Executed Successfully!")
        
        # Display Core Business Metric Metrics[cite: 1]
        st.header("📈 Probabilistic Revenue Planning Horizons")
        st.dataframe(res_df[['forecast_horizon_days', 'expected_aggregate_revenue', 'revenue_lower_bound', 'revenue_upper_bound']])
        
        # Display Target ROAS Intervals[cite: 1]
        st.header("🎯 Blended Target ROAS Outlook")
        st.dataframe(res_df[['forecast_horizon_days', 'expected_blended_roas', 'roas_lower_bound', 'roas_upper_bound']])
        
        # Display AI Causal Inference Summaries Layer[cite: 1]
        st.header("🤖 AI-Generated Supply Chain Causal Context")
        for idx, row in res_df.iterrows():
            with st.expander(f"Context Summary for Day {row['forecast_horizon_days']} Planning Window"):
                st.write(row['ai_causal_summary'])
