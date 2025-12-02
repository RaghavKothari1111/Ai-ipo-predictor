import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestRegressor

DATA_DIR = "data"
MASTER_CSV_PATH = os.path.join(DATA_DIR, "master_ipo_v3.csv")

def audit_model():
    print("Starting Model Brain Audit...")
    
    if not os.path.exists(MASTER_CSV_PATH):
        print("Master dataset not found.")
        return

    df = pd.read_csv(MASTER_CSV_PATH)
    
    # Filter for Listed IPOs
    if 'Listing_Price' not in df.columns:
        print("Listing_Price column missing.")
        return
        
    df = df[df['Listing_Price'] > 0].copy()
    
    if df.empty:
        print("No listed IPOs found to train on.")
        return
        
    # Calculate Target
    # Formula: ((Listing_Price - Issue_Price) / Issue_Price) * 100
    # Filter out Issue_Price = 0
    df = df[df['Issue_Price'] > 0].copy()
    df['Listing_Gain_Percent'] = ((df['Listing_Price'] - df['Issue_Price']) / df['Issue_Price']) * 100
    
    # Features requested by user
    feature_cols = ['Issue_Price', 'GMP', 'IPO_Size_Cr', 'Sub_QIB', 'Sub_NII', 'Sub_Retail', 'Nifty_Trend_30D', 'Current_VIX']
    
    # Ensure columns exist and are numeric
    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0.0
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)
        
    X = df[feature_cols]
    y = df['Listing_Gain_Percent']
    
    print(f"Training on {len(df)} records...")
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    importances = model.feature_importances_
    feature_importance = pd.DataFrame({'Feature': feature_cols, 'Importance': importances})
    feature_importance = feature_importance.sort_values(by='Importance', ascending=False)
    
    print("\n--- Feature Importance ---")
    print(feature_importance.to_string(index=False))
    
    # Verification
    print("\n--- Verification Results ---")
    
    nifty_imp = feature_importance[feature_importance['Feature'] == 'Nifty_Trend_30D']['Importance'].values[0]
    vix_imp = feature_importance[feature_importance['Feature'] == 'Current_VIX']['Importance'].values[0]
    
    if nifty_imp > 0:
        print(f"✅ Model is using Nifty Trend (Importance: {nifty_imp:.4f})")
    else:
        print("⚠️ Model is IGNORING Nifty Trend.")
        
    if vix_imp > 0:
        print(f"✅ Model is using VIX (Importance: {vix_imp:.4f})")
    else:
        print("⚠️ Model is IGNORING VIX.")

if __name__ == "__main__":
    audit_model()
