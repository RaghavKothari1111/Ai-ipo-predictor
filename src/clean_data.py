import pandas as pd
import os

DATA_DIR = "data"
MASTER_CSV_PATH = os.path.join(DATA_DIR, "master_ipo_v3.csv")

def clean_data():
    if not os.path.exists(MASTER_CSV_PATH):
        print(f"File not found: {MASTER_CSV_PATH}")
        return

    print(f"Loading {MASTER_CSV_PATH}...")
    df = pd.read_csv(MASTER_CSV_PATH)
    initial_count = len(df)

    # Ensure Issue_Price is numeric
    if 'Issue_Price' in df.columns:
        df['Issue_Price'] = pd.to_numeric(df['Issue_Price'], errors='coerce').fillna(0.0)
    else:
        print("Column 'Issue_Price' not found. Creating it with 0.0.")
        df['Issue_Price'] = 0.0

    # Filter Logic: Keep rows where Issue_Price > 0
    clean_df = df[df['Issue_Price'] > 0].copy()
    
    final_count = len(clean_df)
    dropped_count = initial_count - final_count

    # Save back
    clean_df.to_csv(MASTER_CSV_PATH, index=False)
    
    print(f"Cleaned data! Dropped {dropped_count} bad rows. Remaining: {final_count} valid IPOs.")

if __name__ == "__main__":
    clean_data()
