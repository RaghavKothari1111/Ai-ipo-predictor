import pandas as pd
import numpy as np
import os
import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

DATA_DIR = "data"
MASTER_CSV_PATH = os.path.join(DATA_DIR, "master_ipo_v3.csv")
MODEL_PATH = os.path.join("models", "ipo_v3_model.pkl")

def train_model():
    if not os.path.exists(MASTER_CSV_PATH):
        print("Master dataset not found.")
        return

    df = pd.read_csv(MASTER_CSV_PATH)
    
    # Filter for rows with known Listing Price > 0 (Fix 2)
    df_train = df.dropna(subset=['Listing_Price']).copy()
    df_train = df_train[df_train['Listing_Price'] > 0]
    
    # Safety Check (Fix 1)
    if len(df_train) < 10:
        print(f"Not enough data to train yet. Found {len(df_train)} valid rows. Accumulating daily data...")
        return

    print(f"Training on {len(df_train)} listed IPOs.")

    # Features
    features = ['GMP', 'GMP_High', 'Sub', 'Has_Anchor', 'Nifty_Trend_7D', 'India_VIX_Close']
    target = 'Listing_Price'
    
    X = df_train[features].fillna(0)
    y = df_train[target]
    
    # Split
    if len(df_train) < 5:
        X_train, X_test, y_train, y_test = X, X, y, y
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
    # Train
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate
    preds = model.predict(X_test)
    mse = mean_squared_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    
    print(f"Model V3 Trained.")
    print(f"MSE: {mse:.2f}")
    print(f"R2 Score: {r2:.2f}")
    
    # Save
    os.makedirs("models", exist_ok=True)
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(model, f)
    print(f"Model saved to {MODEL_PATH}")

if __name__ == "__main__":
    train_model()
