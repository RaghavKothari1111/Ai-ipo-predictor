import pandas as pd
import numpy as np
import os
import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error

DATA_DIR = "data"
MASTER_CSV_PATH = os.path.join(DATA_DIR, "master_ipo_v3.csv")
GMP_HISTORY_CSV_PATH = os.path.join(DATA_DIR, "gmp_history.csv")
MODEL_PATH = "models/ipo_model.pkl"
PREDICTIONS_CSV_PATH = os.path.join(DATA_DIR, "latest_predictions.csv")

def load_data():
    """Loads master data and history data."""
    if not os.path.exists(MASTER_CSV_PATH):
        print("Master dataset not found.")
        return None, None
    
    master_df = pd.read_csv(MASTER_CSV_PATH)
    
    if os.path.exists(GMP_HISTORY_CSV_PATH):
        history_df = pd.read_csv(GMP_HISTORY_CSV_PATH)
        history_df['Date_Time'] = pd.to_datetime(history_df['Date_Time'])
    else:
        history_df = pd.DataFrame(columns=['Date_Time', 'IPO_Name', 'GMP', 'Sub', 'Nifty_Trend', 'VIX'])
        
    return master_df, history_df

def calculate_features(master_df, history_df):
    """Calculates advanced features like GMP Trend and Avg VIX."""
    
    # Initialize new features
    master_df['GMP_Trend'] = 0.0
    master_df['Avg_VIX'] = 12.0 # Default
    
    if history_df.empty:
        return master_df

    for index, row in master_df.iterrows():
        name = row['Name']
        
        # Get history for this IPO
        ipo_history = history_df[history_df['IPO_Name'] == name].sort_values('Date_Time')
        
        if not ipo_history.empty:
            # GMP Trend: Latest GMP - First Recorded GMP
            try:
                first_gmp = float(ipo_history.iloc[0]['GMP']) if pd.notnull(ipo_history.iloc[0]['GMP']) else 0.0
                latest_gmp = float(ipo_history.iloc[-1]['GMP']) if pd.notnull(ipo_history.iloc[-1]['GMP']) else 0.0
                master_df.at[index, 'GMP_Trend'] = latest_gmp - first_gmp
            except:
                master_df.at[index, 'GMP_Trend'] = 0.0
                
            # Avg VIX
            try:
                avg_vix = ipo_history['VIX'].mean()
                if pd.notnull(avg_vix):
                    master_df.at[index, 'Avg_VIX'] = avg_vix
            except:
                pass
                
    return master_df

def train_and_predict():
    print("Starting Continuous Learning Loop (Gain % Prediction)...")
    
    master_df, history_df = load_data()
    if master_df is None or master_df.empty:
        print("No data to process.")
        return

    # Feature Engineering
    print("Calculating features from history...")
    df = calculate_features(master_df, history_df)
    
    # Prepare Data
    feature_cols = ['GMP', 'Sub', 'Nifty_Trend_7D', 'GMP_Trend', 'Avg_VIX']
    
    # Ensure columns are numeric
    for col in feature_cols + ['Listing_Price', 'Issue_Price']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)
            
    # Calculate Target: Listing Gain Percent
    # Formula: ((Listing_Price - Issue_Price) / Issue_Price) * 100
    # Filter out rows where Issue_Price is 0 to avoid division by zero
    df = df[df['Issue_Price'] > 0].copy()
    
    if df.empty:
        print("No valid data with Issue Price > 0.")
        return
        
    df['Listing_Gain_Percent'] = ((df['Listing_Price'] - df['Issue_Price']) / df['Issue_Price']) * 100
    
    # Split Data based on Status
    if 'Status' not in df.columns:
        df['Status'] = df['Listing_Price'].apply(lambda x: 'Listed' if x > 0 else 'Upcoming')
        
    df_train = df[df['Status'] == 'Listed'].copy()
    df_predict = df[df['Status'] == 'Upcoming'].copy()
    
    print(f"Training Data: {len(df_train)} rows. Prediction Data: {len(df_predict)} rows.")
    
    target_col = 'Listing_Gain_Percent'
    
    # --- RETRAIN ---
    if len(df_train) > 5: # Threshold
        print(f"Retraining model on {len(df_train)} listed IPOs...")
        X_train = df_train[feature_cols]
        y_train = df_train[target_col]
        
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # Save Model
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        with open(MODEL_PATH, 'wb') as f:
            pickle.dump(model, f)
        print(f"Model saved to {MODEL_PATH}")
        
        # --- PERFORMANCE REPORT ---
        last_5 = df_train.tail(5)
        if len(last_5) > 0:
            X_test = last_5[feature_cols]
            y_true_gain = last_5[target_col]
            y_pred_gain = model.predict(X_test)
            
            # Convert back to Price for report
            y_true_price = last_5['Listing_Price']
            y_pred_price = last_5['Issue_Price'] * (1 + y_pred_gain / 100)
            
            mape = mean_absolute_percentage_error(y_true_price, y_pred_price) * 100
            print(f"\n--- PERFORMANCE REPORT (Last 5 Listed IPOs) ---")
            for i in range(len(last_5)):
                name = last_5.iloc[i]['Name']
                actual = y_true_price.iloc[i]
                predicted = y_pred_price.iloc[i]
                gain_pred = y_pred_gain[i]
                print(f"IPO: {name} | Actual: {actual:.2f} | Predicted: {predicted:.2f} (Gain: {gain_pred:.1f}%) | Error: {abs(actual-predicted):.2f}")
            print(f"Model Accuracy Updated. Last Error (MAPE): {mape:.2f}%")
            
    else:
        print("Not enough training data. Skipping retraining.")
        if os.path.exists(MODEL_PATH):
            with open(MODEL_PATH, 'rb') as f:
                model = pickle.load(f)
        else:
            print("No existing model found. Cannot predict.")
            return

    # --- PREDICT ---
    if not df_predict.empty:
        print("\n--- PREDICTIONS FOR UPCOMING IPOS ---")
        X_upcoming = df_predict[feature_cols]
        predictions_gain = model.predict(X_upcoming)
        
        results = []
        for i, (index, row) in enumerate(df_predict.iterrows()):
            name = row['Name']
            gmp = row['GMP']
            issue_price = row['Issue_Price']
            pred_gain = predictions_gain[i]
            
            # Calculate Predicted Price
            pred_price = issue_price * (1 + pred_gain / 100)
            
            # Guardrail: GMP Floor
            # If Predicted Price < Issue Price (Negative Gain) BUT GMP is Positive
            # Trust GMP -> Floor = Issue Price + GMP
            if pred_price < issue_price and gmp > 0:
                print(f"  [Guardrail Triggered] {name}: Pred {pred_price:.2f} < Issue {issue_price} but GMP {gmp} > 0. Clamping.")
                pred_price = issue_price + gmp
            
            print(f"[{name}] Issue: {issue_price} | GMP: {gmp} -> Pred Gain: {pred_gain:.1f}% -> Price: ₹{pred_price:.2f}")
            
            results.append({
                'Name': name,
                'Issue_Price': issue_price,
                'Current_GMP': gmp,
                'Predicted_Gain_Percent': pred_gain,
                'Predicted_Listing_Price': pred_price,
                'Prediction_Date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
            })
            
        # Save Predictions
        pred_df = pd.DataFrame(results)
        pred_df.to_csv(PREDICTIONS_CSV_PATH, index=False)
        print(f"Predictions saved to {PREDICTIONS_CSV_PATH}")
    else:
        print("No upcoming IPOs to predict.")

if __name__ == "__main__":
    train_and_predict()
