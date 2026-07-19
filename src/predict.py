import argparse
import pandas as pd
import numpy as np
from google import genai
from google.genai import types

def generate_causal_summary(revenue_forecast, roas_forecast):
    """Utilizes Gemini API to interpret probabilistic anomalies and supply risks[cite: 1]."""
    try:
        # Initializing client via standard environment parameter GEMINI_API_KEY
        client = genai.Client()
        
        prompt = f"""
        Analyze the following Ayurvedic E-commerce marketing and supply chain predictions:
        - 90-Day Expected Cumulative Revenue Range: ${revenue_forecast['lower_bound']:.2f} to ${revenue_forecast['upper_bound']:.2f}
        - Projected Blended Target ROAS Range: {roas_forecast['lower_bound']:.2f}x to {roas_forecast['upper_bound']:.2f}x
        
        Generate a concise operational summary identifying potential risks related to traditional herb inventory constraints (e.g., Ashwagandha/Neem transit periods) and marketing spend adjustments.
        """
        
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt,
        )
        return response.text.strip()
    except Exception as e:
        return f"Causal AI layer bypass: Summary generation omitted due to environment configuration ({str(e)})."

def run_prediction(features_path, model_path, output_path):
    features = pd.read_parquet(features_path)
    
    # Simulating standard forecasting window periods[cite: 1]
    horizons = [30, 60, 90]
    output_records = []
    
    # Baseline forecasting values derived dynamically from features
    base_rev = features['historical_revenue'].iloc[-1] if not features.empty else 15000
    base_spend = features['total_marketing_spend'].iloc[-1] if not features.empty else 5000
    avg_delay = features['warehouse_delay_days'].mean() if not features.empty else 2
    
    for days in horizons:
        # Creating realistic probabilistic variations instead of single values[cite: 1]
        # Supply disruptions scale the uncertainty window upward
        uncertainty_factor = 0.05 * (days / 30) * (1 + (avg_delay * 0.1))
        
        expected_rev = base_rev * (1 + (0.02 * (days / 30)))
        expected_roas = (expected_rev / base_spend) if base_spend > 0 else 3.0
        
        rev_range = {
            'expected': expected_rev,
            'lower_bound': expected_rev * (1 - uncertainty_factor),
            'upper_bound': expected_rev * (1 + uncertainty_factor)
        }
        
        roas_range = {
            'expected': expected_roas,
            'lower_bound': expected_roas * (1 - uncertainty_factor),
            'upper_bound': expected_roas * (1 + uncertainty_factor)
        }
        
        # Trigger the causal analysis layer for the aggregate timeframe[cite: 1]
        ai_summary = generate_causal_summary(rev_range, roas_range)
        
        output_records.append({
            'forecast_horizon_days': days,
            'expected_aggregate_revenue': round(rev_range['expected'], 2),
            'revenue_lower_bound': round(rev_range['lower_bound'], 2),
            'revenue_upper_bound': round(rev_range['upper_bound'], 2),
            'expected_blended_roas': round(roas_range['expected'], 2),
            'roas_lower_bound': round(roas_range['lower_bound'], 2),
            'roas_upper_bound': round(roas_range['upper_bound'], 2),
            'ai_causal_summary': ai_summary
        })
        
    predictions_df = pd.DataFrame(output_records)
    predictions_df.to_csv(output_path, index=False)[cite: 2]
    print(f"Probabilistic tracking engine outputs stored at {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    
    run_prediction(args.features, args.model, args.output)
