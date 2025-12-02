import os
import pickle
import time
from datetime import datetime, timedelta
import sys

MODEL_PATH = "models/ipo_model.pkl"

def inspect_model():
    print("Starting Model Forensics...")
    
    # 1. Timestamp Check
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Model file not found at {MODEL_PATH}")
        return

    mod_time = os.path.getmtime(MODEL_PATH)
    mod_datetime = datetime.fromtimestamp(mod_time)
    now = datetime.now()
    diff = now - mod_datetime
    minutes_ago = diff.total_seconds() / 60
    hours_ago = minutes_ago / 60
    
    print(f"Model file last modified: {mod_datetime.strftime('%Y-%m-%d %H:%M:%S')} ({minutes_ago:.1f} minutes ago)")
    
    is_fresh = minutes_ago < 240 # 4 hours
    
    # 2. Brain Scan
    try:
        with open(MODEL_PATH, 'rb') as f:
            model = pickle.load(f)
            
        if hasattr(model, 'feature_names_in_'):
            features = list(model.feature_names_in_)
            print(f"\nFeatures detected in model ({len(features)}):")
            print(features)
            
            has_vix = "Current_VIX" in features
            has_trend = "Nifty_Trend_30D" in features
            has_qib = "Sub_QIB" in features
            
            # Verdict
            print("\n--- VERDICT ---")
            if is_fresh and has_vix and has_trend:
                print("✅ PASS: Model is fresh and contains Market Sentiment features.")
            elif is_fresh and not (has_vix and has_trend):
                print("❌ FAIL (Old Code): Model is fresh but MISSING new features.")
            elif not is_fresh:
                print("❌ FAIL (Stale File): Model file is old (> 4 hours). Training might have failed or not run.")
                
        else:
            print("\n⚠️ Model does not have 'feature_names_in_' attribute. It might be an older scikit-learn version or not a standard estimator.")
            
    except Exception as e:
        print(f"\n❌ Error loading model: {e}")

if __name__ == "__main__":
    inspect_model()
