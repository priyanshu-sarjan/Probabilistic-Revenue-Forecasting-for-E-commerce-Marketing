import os
import argparse
import pandas as pd
import numpy as np

def build_features(data_dir, output_file):
    # Dynamically find files in the directory[cite: 2]
    files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
    if not files:
        raise FileNotFoundError(f"No CSV data found in target directory: {data_dir}")
    
    # Read the first available matching dataset[cite: 2]
    raw_path = os.path.join(data_dir, files[0])
    df = pd.read_csv(raw_path)
    
    # Core Feature Engineering combining Ad spends and supply traceability metrics
    features = pd.DataFrame()
    features['date'] = pd.to_datetime(df['date'])
    
    # Aggregating channel spend fields
    features['total_marketing_spend'] = (
        df.get('google_ads_spend', 0) + 
        df.get('meta_ads_spend', 0) + 
        df.get('ms_ads_spend', 0)
    )
    
    # Splicing trace elements like transit delays & herb batch purity metrics
    features['harvest_seasonality_index'] = df.get('harvest_seasonality_index', 1.0)
    features['warehouse_delay_days'] = df.get('warehouse_delay_days', 0).fillna(0)
    features['batch_purity_score'] = df.get('batch_purity_score', 95.0)
    
    # Historic performance baseline metrics[cite: 1]
    features['historical_revenue'] = df.get('revenue', 0)
    features['historical_roas'] = df.get('roas', 0)
    
    # Save processed features locally[cite: 2]
    features.to_parquet(output_file, index=False)
    print(f"Successfully processed {len(features)} records into {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    build_features(args.data_dir, args.out)
