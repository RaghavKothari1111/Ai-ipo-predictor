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
    
    # Initialize new features if not present
    if 'GMP_Trend' not in master_df.columns:
        master_df['GMP_Trend'] = 0.0
    
    # Note: Avg_VIX might be replaced by Current_VIX from hybrid collector, 
    # but we can still calculate Avg_VIX from history if needed.
    # The hybrid collector adds 'Current_VIX'.
    
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
                
    return master_df

def train_and_predict():
    print("Starting Continuous Learning Loop (Hybrid Data)...")
    
    master_df, history_df = load_data()
    if master_df is None or master_df.empty:
        print("No data to process.")
        return

    # Feature Engineering
    print("Calculating features from history...")
    df = calculate_features(master_df, history_df)
    
    # Prepare Data
    # New Feature Set: Issue_Price, GMP, IPO_Size_Cr, Sub_QIB, Sub_NII, Sub_Retail, Nifty_Trend_30D, Current_VIX
    feature_cols = ['GMP', 'IPO_Size_Cr', 'Sub_QIB', 'Sub_NII', 'Sub_Retail', 'Nifty_Trend_30D', 'Current_VIX', 'GMP_Trend']
    
    # Ensure columns are numeric
    for col in feature_cols + ['Listing_Price', 'Issue_Price']:
        if col not in df.columns:
            df[col] = 0.0
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
        # We still run model prediction for all, but might override it
        predictions_gain_model = model.predict(X_upcoming)
        
        results = []
        for i, (index, row) in enumerate(df_predict.iterrows()):
            name = row['Name']
            issue_price = row['Issue_Price']
            gmp = row['GMP']
            sub_qib = row['Sub_QIB']
            
            # Conditional Logic based on Data Stage
            # We check Data_Stage if available, otherwise fallback to QIB check
            data_stage = row.get('Data_Stage', 'Early')
            
            # If Data_Stage is missing or NaN, infer from QIB
            if pd.isna(data_stage) or data_stage == 0:
                data_stage = "Mature" if sub_qib > 0 else "Early"
            
            if data_stage == "Mature" or sub_qib > 0:
                # Scenario A: Mature Data -> Use AI Model
                pred_gain = predictions_gain_model[i]
                prediction_type = "AI_Model"
                method = "AI_Model"
                
                # QIB Boost Logic (Only for AI Model)
                if sub_qib > 50:
                    print(f"  [QIB Boost] {name}: QIB {sub_qib}x > 50x. Boosting gain by 5%.")
                    pred_gain += 5.0
                
                # Calculate Predicted Price
                pred_price = issue_price * (1 + pred_gain / 100)
                
                # Guardrail: GMP Floor (Safety Check)
                if pred_price < issue_price and gmp > 0:
                    print(f"  [Guardrail Triggered] {name}: Pred {pred_price:.2f} < Issue {issue_price} but GMP {gmp} > 0. Clamping.")
                    pred_price = issue_price + gmp
                    # Recalculate gain based on clamped price
                    pred_gain = ((pred_price - issue_price) / issue_price) * 100
                    
            else:
                # Scenario B: Early Stage -> Use GMP Only
                prediction_type = "GMP_Fallback"
                method = "GMP_Fallback"
                pred_price = issue_price + gmp
                if issue_price > 0:
                    pred_gain = (gmp / issue_price) * 100
                else:
                    pred_gain = 0.0
                
                print(f"  [Early Stage] {name}: Data Stage is '{data_stage}'. Using GMP-based prediction.")

            print(f"[{name}] Issue: {issue_price} | QIB: {sub_qib}x | Method: {method} | Pred Gain: {pred_gain:.1f}% -> Final Price: ₹{pred_price:.2f}")
            
            results.append({
                'Name': name,
                'Issue_Price': issue_price,
                'Predicted_Gain_Percent': pred_gain,
                'Predicted_Final_Price': pred_price,
                'Current_GMP': gmp,
                'Sub_QIB': sub_qib,
                'Method': method,
                'Prediction_Date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
            })
            
        # Save Predictions
        pred_df = pd.DataFrame(results)
        cols = ['Name', 'Issue_Price', 'Predicted_Gain_Percent', 'Predicted_Final_Price', 'Current_GMP', 'Sub_QIB', 'Method', 'Prediction_Date']
        pred_df = pred_df[cols]
        
        pred_df.to_csv(PREDICTIONS_CSV_PATH, index=False)
        print(f"Predictions saved to {PREDICTIONS_CSV_PATH}")
    else:
        print("No upcoming IPOs to predict.")

if __name__ == "__main__":
    train_and_predict()
