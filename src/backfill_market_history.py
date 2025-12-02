import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import os

DATA_DIR = "data"
MASTER_CSV_PATH = os.path.join(DATA_DIR, "master_ipo_v3.csv")

def backfill_history():
    print("Starting Historical Market Data Backfill...")
    
    if not os.path.exists(MASTER_CSV_PATH):
        print("Master dataset not found.")
        return

    df = pd.read_csv(MASTER_CSV_PATH)
    
    # Filter for Listed IPOs or rows with Listing_Date
    # We need Listing_Date to be present and not NaN
    mask = df['Listing_Date'].notna() & (df['Listing_Date'] != '')
    listed_indices = df[mask].index
    
    print(f"Found {len(listed_indices)} IPOs with listing dates to process.")
    
    updated_count = 0
    
    for idx in listed_indices:
        row = df.loc[idx]
        listing_date_str = str(row['Listing_Date'])
        name = row['Name']
        
        try:
            listing_date = datetime.strptime(listing_date_str, "%Y-%m-%d")
        except ValueError:
            print(f"Skipping {name}: Invalid date format {listing_date_str}")
            continue
            
        # Skip if listing date is in the future (Upcoming)
        if listing_date > datetime.now():
            continue
            
        print(f"Processing {name} (Listed: {listing_date_str})...")
        
        # Define Date Range
        # We need data for Listing Date and 30 days prior
        start_date = listing_date - timedelta(days=40)
        end_date = listing_date + timedelta(days=1)
        
        try:
            # Fetch Data
            # Suppress progress bar
            nifty = yf.download("^NSEI", start=start_date, end=end_date, progress=False)
            vix = yf.download("^INDIAVIX", start=start_date, end=end_date, progress=False)
            
            # 1. Historic VIX (Close on Listing Date)
            if not vix.empty:
                # Get VIX on listing date or nearest previous trading day
                try:
                    # Use asof to find nearest date
                    idx_vix = vix.index.get_indexer([listing_date], method='nearest')[0]
                    historic_vix = float(vix['Close'].iloc[idx_vix])
                    df.at[idx, 'Current_VIX'] = historic_vix
                except:
                    print(f"  - VIX data missing for {listing_date_str}")
            
            # 2. Historic Trend (Price on Listing vs 30 Days Ago)
            if not nifty.empty:
                try:
                    # Price on Listing Date
                    idx_now = nifty.index.get_indexer([listing_date], method='nearest')[0]
                    price_now = float(nifty['Close'].iloc[idx_now])
                    
                    # Price 30 Days Ago
                    date_30d_ago = listing_date - timedelta(days=30)
                    idx_old = nifty.index.get_indexer([date_30d_ago], method='nearest')[0]
                    price_old = float(nifty['Close'].iloc[idx_old])
                    
                    if price_old > 0:
                        trend = (price_now - price_old) / price_old
                        df.at[idx, 'Nifty_Trend_30D'] = trend
                except:
                    print(f"  - Nifty data missing for trend calc")
            
            updated_count += 1
            
        except Exception as e:
            print(f"  - Error fetching data: {e}")
            
    # Save
    df.to_csv(MASTER_CSV_PATH, index=False)
    print(f"\nBackfill Complete. Updated {updated_count} records.")
    
    # Verify
    print("\n--- Verification Sample (First 3 Listed Updated) ---")
    # Show rows where Status is Listed (or we updated them)
    # We can just show the rows we processed if we kept track, or just filter again
    sample = df[mask].head(3)[['Name', 'Listing_Date', 'Current_VIX', 'Nifty_Trend_30D']]
    print(sample.to_string(index=False))

if __name__ == "__main__":
    backfill_history()
