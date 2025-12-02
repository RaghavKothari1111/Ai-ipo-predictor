import requests
import pandas as pd
import os
import re
import yfinance as yf
from datetime import datetime, timedelta
from bs4 import BeautifulSoup
import sys

# Configuration
DATA_DIR = "data"
MASTER_CSV_PATH = os.path.join(DATA_DIR, "master_ipo_v3.csv")
GMP_HISTORY_CSV_PATH = os.path.join(DATA_DIR, "gmp_history.csv")

# API Templates
INVESTORGAIN_URL_TEMPLATE = "https://webnodejs.investorgain.com/cloud/report/data-read/331/{day}/{month}/{year}/{fy}/0/all?search=&v=09-18"
CHITTORGARH_URL_TEMPLATE = "https://webnodejs.chittorgarh.com/cloud/report/data-read/22/{day}/{month}/{year}/{fy}/0/0/0?search=&v=08-39"

def get_financial_year(date_obj):
    """Calculates Indian Financial Year (e.g., 2025-26)."""
    if date_obj.month >= 4:
        start_year = date_obj.year
        end_year = (date_obj.year + 1) % 100
    else:
        start_year = date_obj.year - 1
        end_year = date_obj.year % 100
    return f"{start_year}-{end_year}"

def clean_html(raw_html):
    """Removes HTML tags and decodes entities."""
    if not isinstance(raw_html, str):
        return str(raw_html)
    soup = BeautifulSoup(raw_html, "html.parser")
    return soup.get_text(strip=True)

def parse_name(raw_name):
    """Normalizes company name."""
    if not raw_name: return "Unknown"
    name_clean = clean_html(raw_name)
    normalized = re.sub(r'(?:BSE|NSE)\s*SME.*', '', name_clean).strip()
    return normalized

def parse_currency(val):
    """Transforms ₹--(0.00%) to 0.0"""
    if not val or val == '-': return 0.0
    val = clean_html(val)
    match = re.search(r'₹?([\d\.]+)', val)
    if match:
        return float(match.group(1))
    return 0.0

def parse_date(date_str):
    """Parses date like '12-Dec' to 'YYYY-MM-DD'."""
    if not date_str or date_str == '-': return None
    try:
        return datetime.strptime(date_str, "%Y-%m-%d").strftime("%Y-%m-%d")
    except:
        pass
    try:
        dt = datetime.strptime(date_str + "-2025", "%d-%b-%Y")
        return dt.strftime("%Y-%m-%d")
    except:
        return None

def fetch_json(url_template):
    """Generic JSON fetcher."""
    now = datetime.now()
    day = now.day
    month = now.month
    year = now.year
    fy = get_financial_year(now)
    
    url = url_template.format(day=day, month=month, year=year, fy=fy)
    print(f"Fetching from: {url}")
    
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        return data.get("reportTableData", [])
    except Exception as e:
        print(f"Error fetching data: {e}")
        return []

def get_market_data():
    """Fetches Nifty 30D Trend and Current VIX."""
    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=40) # Fetch enough for 30D lookback
        
        # Suppress yfinance progress
        nifty = yf.download("^NSEI", start=start_date, end=end_date + timedelta(days=1), progress=False)
        vix = yf.download("^INDIAVIX", start=start_date, end=end_date + timedelta(days=1), progress=False)
        
        # VIX
        if vix.empty:
            current_vix = 12.0
        else:
            current_vix = float(vix['Close'].iloc[-1])
            if pd.isna(current_vix) or current_vix == 0.0:
                current_vix = 12.0
                
        # Nifty Trend 30D
        if nifty.empty:
            nifty_trend_30d = 0.0
        else:
            current_price = float(nifty['Close'].iloc[-1])
            
            # Find price ~30 days ago
            target_date_30d = nifty.index[-1] - timedelta(days=30)
            try:
                idx = nifty.index.get_indexer([target_date_30d], method='nearest')[0]
                price_30d_ago = float(nifty['Close'].iloc[idx])
                nifty_trend_30d = (current_price - price_30d_ago) / price_30d_ago
            except:
                nifty_trend_30d = 0.0
                
        return nifty_trend_30d, current_vix
        
    except Exception as e:
        print(f"Market Data Error: {e}")
        return 0.0, 12.0

def process_investorgain(data):
    """Process Source 1 (Hype) Data."""
    processed = []
    for row in data:
        raw_name = row.get("Name", "")
        
        # Extract Listing Price (Target)
        name_clean_for_extraction = clean_html(raw_name)
        match = re.search(r'L@([\d\.]+)', name_clean_for_extraction)
        listing_price = float(match.group(1)) if match else None
        
        # Normalize Name
        name = parse_name(raw_name)
        
        # Extract Issue Price (Input)
        raw_price = clean_html(row.get("Price", ""))
        issue_price = 0.0
        if raw_price and raw_price != '-':
            try:
                if '-' in raw_price:
                    parts = raw_price.split('-')
                    issue_price = float(re.sub(r'[^\d\.]', '', parts[-1]))
                else:
                    issue_price = float(re.sub(r'[^\d\.]', '', raw_price))
            except:
                issue_price = 0.0
        
        # GMP
        gmp = parse_currency(row.get("GMP", ""))
        
        # Dates
        raw_listing = row.get("~Str_Listing", "") or row.get("Listing", "")
        listing_date = parse_date(raw_listing)
        
        # Status
        if listing_price is not None and listing_price > 0:
            status = "Listed"
        else:
            status = "Upcoming"
            
        processed.append({
            "Name": name,
            "Issue_Price": issue_price,
            "GMP": gmp,
            "Listing_Price": listing_price,
            "Listing_Date": listing_date,
            "Status": status
        })
    return pd.DataFrame(processed)

def process_chittorgarh(data):
    """Process Source 2 (Fundamental) Data."""
    processed = []
    for row in data:
        raw_name = row.get("Name", "")
        name = parse_name(raw_name)
        
        # Issue Size
        try:
            size_str = clean_html(row.get("Issue Size", "")).replace(',', '')
            issue_size_cr = float(re.search(r'([\d\.]+)', size_str).group(1)) if re.search(r'([\d\.]+)', size_str) else 0.0
        except:
            issue_size_cr = 0.0
            
        # Subscriptions
        def parse_sub(key):
            try:
                val = clean_html(row.get(key, "")).replace('x', '').replace(',', '')
                return float(val)
            except:
                return 0.0
        
        sub_qib = parse_sub("QIB")
        sub_nii = parse_sub("NII")
        sub_retail = parse_sub("Retail")
        
        processed.append({
            "Name": name,
            "IPO_Size_Cr": issue_size_cr,
            "Sub_QIB": sub_qib,
            "Sub_NII": sub_nii,
            "Sub_Retail": sub_retail
        })
    return pd.DataFrame(processed)

def main():
    print("Starting Hybrid Data Collection...")
    
    # 1. Fetch Data
    print("\n--- Fetching Source 1: Investorgain (Hype) ---")
    data_hype = fetch_json(INVESTORGAIN_URL_TEMPLATE)
    df_hype = process_investorgain(data_hype)
    print(f"Fetched {len(df_hype)} records from Investorgain.")
    
    print("\n--- Fetching Source 2: Chittorgarh (Fundamentals) ---")
    data_fund = fetch_json(CHITTORGARH_URL_TEMPLATE)
    df_fund = process_chittorgarh(data_fund)
    print(f"Fetched {len(df_fund)} records from Chittorgarh.")
    
    # 2. Smart Merge
    print("\n--- Merging Datasets ---")
    # Left join on Name (Hype is primary because it has GMP and Status)
    merged_df = pd.merge(df_hype, df_fund, on="Name", how="left")
    
    # Fill NaN for missing fundamental data
    cols_to_fill = ['IPO_Size_Cr', 'Sub_QIB', 'Sub_NII', 'Sub_Retail']
    for col in cols_to_fill:
        if col in merged_df.columns:
            merged_df[col] = merged_df[col].fillna(0.0)
            
    print(f"Merged Data Shape: {merged_df.shape}")
    
    # 3. Market Sentiment
    print("\n--- Fetching Market Sentiment ---")
    nifty_trend_30d, current_vix = get_market_data()
    print(f"Nifty Trend (30D): {nifty_trend_30d:.4f}, VIX: {current_vix}")
    
    merged_df['Nifty_Trend_30D'] = nifty_trend_30d
    merged_df['Avg_VIX'] = current_vix # Using current VIX as proxy for Avg VIX for now, or we can keep Avg_VIX logic in trainer
    # Actually, the prompt says "Add columns Nifty_Trend_30D and Current_VIX".
    merged_df['Current_VIX'] = current_vix
    
    # 4. Upsert to Master CSV
    print("\n--- Updating Master Dataset ---")
    os.makedirs(DATA_DIR, exist_ok=True)
    
    if os.path.exists(MASTER_CSV_PATH):
        try:
            master_df = pd.read_csv(MASTER_CSV_PATH)
            
            # Clean Master Name just in case
            master_df['Name'] = master_df['Name'].astype(str).apply(
                lambda x: re.sub(r'(?:BSE|NSE)\s*SME.*', '', x).strip()
            )
            
            # Ensure new columns exist in master_df
            new_cols = ['IPO_Size_Cr', 'Sub_QIB', 'Sub_NII', 'Sub_Retail', 'Nifty_Trend_30D', 'Current_VIX']
            for col in new_cols:
                if col not in master_df.columns:
                    master_df[col] = 0.0
            
            # Create dict for upsert
            master_dict = {row['Name']: row for _, row in master_df.iterrows()}
            
            # Upsert new data
            for _, row in merged_df.iterrows():
                # We update the row. New columns will be added.
                # If row exists, update it. If not, add it.
                # Note: This overwrites existing data for that IPO with new data.
                # This is correct for "latest status".
                
                # Convert row to dict
                row_dict = row.to_dict()
                
                if row['Name'] in master_dict:
                    # Update existing
                    master_dict[row['Name']].update(row_dict)
                else:
                    # Add new
                    master_dict[row['Name']] = row_dict
            
            final_df = pd.DataFrame(list(master_dict.values()))
            
        except Exception as e:
            print(f"Error reading master csv: {e}. Overwriting.")
            final_df = merged_df
    else:
        final_df = merged_df
        
    # Ensure all columns exist (fill NaNs)
    final_df.fillna(0.0, inplace=True)
    
    final_df.to_csv(MASTER_CSV_PATH, index=False)
    print(f"Saved {len(final_df)} records to {MASTER_CSV_PATH}")
    
    # 5. Alert System
    print("\n--- Missing Data Report ---")
    upcoming_missing_qib = final_df[
        (final_df['Status'] == 'Upcoming') & 
        (final_df['Sub_QIB'] == 0.0)
    ]
    
    if not upcoming_missing_qib.empty:
        print("The following Upcoming IPOs are missing QIB data:")
        for name in upcoming_missing_qib['Name'].unique():
            print(f"- {name}")
    else:
        print("All upcoming IPOs have QIB data.")
        
    # 6. Trigger Training
    print("\n--- Triggering Training ---")
    exit_code = os.system("python3 src/train_model.py")
    if exit_code != 0:
        print("Training script failed.")
    else:
        print("Training completed.")

if __name__ == "__main__":
    main()
