import requests
import pandas as pd
import os
import re
import yfinance as yf
from datetime import datetime, timedelta
from bs4 import BeautifulSoup
import sys
import difflib

# Configuration
DATA_DIR = "data"
MASTER_CSV_PATH = os.path.join(DATA_DIR, "master_ipo_v3.csv")
GMP_HISTORY_CSV_PATH = os.path.join(DATA_DIR, "gmp_history.csv")

# API Templates
INVESTORGAIN_URL_TEMPLATE = "https://webnodejs.investorgain.com/cloud/report/data-read/331/{day}/{month}/{year}/{fy}/0/all?search=&v=09-18"
CHITTORGARH_SME_URL_TEMPLATE = "https://webnodejs.chittorgarh.com/cloud/report/data-read/22/{day}/{month}/{year}/{fy}/0/0/0?search=&v=08-39"
CHITTORGARH_MAIN_URL_TEMPLATE = "https://webnodejs.chittorgarh.com/cloud/report/data-read/21/{day}/{month}/{year}/{fy}/0/0/0?search=&v=11-25"

SECTOR_KEYWORDS = ["Solar", "Energy", "Tech", "Defence", "Green"]

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
    normalized = re.sub(r'(?:BSE|NSE)\s*SME.*', '', name_clean, flags=re.IGNORECASE).strip()
    # Remove IPO suffixes like IPOU, IPOO, IPOCT, IPO, and specifically IPOL@...
    normalized = re.sub(r'\s+IPO[A-Z]*(@.*)?(\(.*\))?$', '', normalized, flags=re.IGNORECASE).strip()
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
        raw_name = row.get("Company Name", "")
        name = parse_name(raw_name)
        
        # Issue Size
        try:
            size_str = clean_html(row.get("Total Issue Amount (Incl.Firm reservations) (Rs.cr.)", "")).replace(',', '')
            issue_size_cr = float(re.search(r'([\d\.]+)', size_str).group(1)) if re.search(r'([\d\.]+)', size_str) else 0.0
        except:
            issue_size_cr = 0.0
            
        # Subscriptions
        def parse_sub(key):
            try:
                val = clean_html(row.get(key, ""))
                # Handle Tilde (~) and Dash (-) for delayed data
                if '~' in val or '-' in val:
                    return 0.0
                val = val.replace('x', '').replace(',', '')
                return float(val)
            except:
                return 0.0
        
        sub_qib = parse_sub("QIB (x)")
        sub_nii = parse_sub("NII (x)")
        sub_retail = parse_sub("Retail (x)")
        
        # Data Stage Logic
        data_stage = "Mature" if sub_qib > 0 else "Early"
        
        processed.append({
            "Name": name,
            "IPO_Size_Cr": issue_size_cr,
            "Sub_QIB": sub_qib,
            "Sub_NII": sub_nii,
            "Sub_Retail": sub_retail,
            "Data_Stage": data_stage
        })
    return pd.DataFrame(processed)

def calculate_gmp_momentum(name, current_gmp, history_df):
    """Calculates GMP Momentum: Current GMP - GMP 3 days ago (Sparse Safe)."""
    if history_df.empty:
        return 0.0
        
    # Filter history for this IPO
    ipo_history = history_df[history_df['IPO_Name'] == name].copy()
    
    if ipo_history.empty:
        return 0.0
        
    # Ensure datetime
    if not pd.api.types.is_datetime64_any_dtype(ipo_history['Date_Time']):
        ipo_history['Date_Time'] = pd.to_datetime(ipo_history['Date_Time'])
        
    ipo_history = ipo_history.sort_values('Date_Time')
    
    # Target: 3 days ago
    target_date = datetime.now() - timedelta(days=3)
    
    # Find records ON or BEFORE target date
    past_records = ipo_history[ipo_history['Date_Time'] <= target_date]
    
    if not past_records.empty:
        # Take the LAST record from the past (closest to the target date from the past)
        # This represents the state of the IPO 3 days ago.
        historic_row = past_records.iloc[-1]
        historic_gmp = float(historic_row['GMP'])
        return current_gmp - historic_gmp
    else:
        # If no history old enough (e.g. IPO added yesterday), use the FIRST record
        # This gives momentum since tracking began
        historic_row = ipo_history.iloc[0]
        historic_gmp = float(historic_row['GMP'])
        return current_gmp - historic_gmp

def check_sector_bonus(name):
    """Checks if name contains hype keywords."""
    for keyword in SECTOR_KEYWORDS:
        if keyword.lower() in name.lower():
            return 1
    return 0

def main():
    print("Starting Hybrid Data Collection v4...")
    
    # Load History for Momentum Calculation
    if os.path.exists(GMP_HISTORY_CSV_PATH):
        history_df = pd.read_csv(GMP_HISTORY_CSV_PATH)
        history_df['Date_Time'] = pd.to_datetime(history_df['Date_Time'])
    else:
        history_df = pd.DataFrame(columns=['Date_Time', 'IPO_Name', 'GMP', 'Sub', 'Nifty_Trend', 'VIX'])
    
    # 1. Fetch Data
    print("\n--- Fetching Source 1: Investorgain (Hype) ---")
    data_hype = fetch_json(INVESTORGAIN_URL_TEMPLATE)
    df_hype = process_investorgain(data_hype)
    print(f"Fetched {len(df_hype)} records from Investorgain (Master Trigger).")
    
    print("\n--- Fetching Source 2: Chittorgarh (Fundamentals) ---")
    # Fetch SME
    print("Fetching SME Data...")
    data_sme = fetch_json(CHITTORGARH_SME_URL_TEMPLATE)
    df_sme = process_chittorgarh(data_sme)
    df_sme['Source_Type'] = 'SME'
    print(f"Fetched {len(df_sme)} SME records.")
    
    # Fetch Mainboard
    print("Fetching Mainboard Data...")
    data_main = fetch_json(CHITTORGARH_MAIN_URL_TEMPLATE)
    df_main = process_chittorgarh(data_main)
    df_main['Source_Type'] = 'Mainboard'
    print(f"Fetched {len(df_main)} Mainboard records.")
    
    # Combine and Deduplicate
    df_fund = pd.concat([df_sme, df_main], ignore_index=True)
    df_fund = df_fund.drop_duplicates(subset=['Name'], keep='last')
    print(f"Total Fundamental Records: {len(df_fund)}")
    
    # 2. Smart Merge & Feature Engineering
    print("\n--- Merging Datasets & Feature Engineering ---")
    
    # Fetch Live Market Data
    print("Fetching Live Market Data...")
    live_nifty_trend, live_vix = get_market_data()
    print(f"Live Nifty Trend: {live_nifty_trend:.4f}, Live VIX: {live_vix}")
    
    # Convert fund df to dict for lookup
    fund_records = df_fund.to_dict('records')
    fund_dict = {row['Name']: row for row in fund_records}
    fund_names = list(fund_dict.keys())
    
    merged_rows = []
    
    matched_mainboard = 0
    matched_sme = 0
    sector_hype_count = 0
    
    for _, hype_row in df_hype.iterrows():
        hype_name = hype_row['Name']
        
        # Try exact match
        if hype_name in fund_dict:
            fund_data = fund_dict[hype_name]
        else:
            # Try fuzzy match
            matches = difflib.get_close_matches(hype_name, fund_names, n=1, cutoff=0.5)
            if matches:
                fund_data = fund_dict[matches[0]]
            else:
                fund_data = {} # No fundamental data found
        
        # Track Source
        if fund_data:
            source_type = fund_data.get('Source_Type', 'Unknown')
            if source_type == 'Mainboard':
                matched_mainboard += 1
            elif source_type == 'SME':
                matched_sme += 1

        # Combine data
        combined = hype_row.to_dict()
        combined['IPO_Size_Cr'] = fund_data.get('IPO_Size_Cr', 0.0)
        combined['Sub_QIB'] = fund_data.get('Sub_QIB', 0.0)
        combined['Sub_NII'] = fund_data.get('Sub_NII', 0.0)
        combined['Sub_Retail'] = fund_data.get('Sub_Retail', 0.0)
        combined['Data_Stage'] = fund_data.get('Data_Stage', 'Early') # Default to Early if missing
        
        # Add Live Market Data (Will be preserved for Listed IPOs later)
        combined['Nifty_Trend_30D'] = live_nifty_trend
        combined['Current_VIX'] = live_vix
        
        # --- Advanced Features ---
        # GMP Momentum
        combined['GMP_Momentum'] = calculate_gmp_momentum(hype_name, combined['GMP'], history_df)
        
        # Sector Bonus
        sector_bonus = check_sector_bonus(hype_name)
        combined['Sector_Bonus'] = sector_bonus
        if sector_bonus == 1:
            sector_hype_count += 1
            
        merged_rows.append(combined)
        
    merged_df = pd.DataFrame(merged_rows)
    print(f"Merged Data Shape: {merged_df.shape}")
    print(f"Enrichment Report: Matched QIB data for {matched_mainboard} Mainboard and {matched_sme} SME IPOs.")
    print(f"Sector Hype detected in {sector_hype_count} stocks.")
    
    # 3. Save & History
    print("\n--- Updating Master Dataset & History ---")
    os.makedirs(DATA_DIR, exist_ok=True)
    
    # Upsert Logic
    if os.path.exists(MASTER_CSV_PATH):
        try:
            master_df = pd.read_csv(MASTER_CSV_PATH)
            
            # Clean Master Name
            master_df['Name'] = master_df['Name'].astype(str).apply(
                lambda x: re.sub(r'(?:BSE|NSE)\s*SME.*', '', x).strip()
            )
            
            # Ensure new columns exist in master_df
            new_cols = ['IPO_Size_Cr', 'Sub_QIB', 'Sub_NII', 'Sub_Retail', 'Data_Stage', 'GMP_Momentum', 'Sector_Bonus', 'Nifty_Trend_30D', 'Current_VIX']
            for col in new_cols:
                if col not in master_df.columns:
                    if col == 'Data_Stage':
                        master_df[col] = 'Early'
                    else:
                        master_df[col] = 0.0
            
            master_dict = master_df.set_index('Name').to_dict('index')
            for name, row_data in master_dict.items():
                row_data['Name'] = name
            
            # Upsert
            for _, row in merged_df.iterrows():
                row_dict = row.to_dict()
                name = row['Name']
                
                if name in master_dict:
                    existing_row = master_dict[name]
                    
                    # 1. Preserve Fundamental Data if new is 0.0
                    preserve_cols = ['IPO_Size_Cr', 'Sub_QIB', 'Sub_NII', 'Sub_Retail']
                    for col in preserve_cols:
                        new_val = float(row_dict.get(col, 0.0))
                        old_val = float(existing_row.get(col, 0.0))
                        if new_val == 0.0 and old_val > 0.0:
                            row_dict[col] = old_val
                            
                    # 2. Preserve Historical Market Data if Listed
                    # If status is Listed, we do NOT want to overwrite VIX/Trend with today's live data
                    # We want to keep the value that was set when it listed (or backfilled)
                    if existing_row.get('Status') == 'Listed':
                        # Check if we have valid historical data
                        old_vix = float(existing_row.get('Current_VIX', 0.0))
                        if old_vix > 0:
                            row_dict['Current_VIX'] = old_vix
                            row_dict['Nifty_Trend_30D'] = existing_row.get('Nifty_Trend_30D', 0.0)
                            # print(f"  [Preserving History] {name}: Keeping VIX={old_vix}")
                        
                        # 3. Preserve Status if new is Upcoming (prevent regression)
                        if row_dict.get('Status') == 'Upcoming':
                            row_dict['Status'] = 'Listed'
                            row_dict['Listing_Price'] = existing_row.get('Listing_Price', 0.0)
                            row_dict['Listing_Date'] = existing_row.get('Listing_Date', None)
                            
                    master_dict[name].update(row_dict)
                else:
                    master_dict[name] = row_dict
            
            final_df = pd.DataFrame(list(master_dict.values()))
            
        except Exception as e:
            print(f"Error reading master csv: {e}. Overwriting.")
            final_df = merged_df
    else:
        final_df = merged_df
        
    final_df.fillna(0.0, inplace=True)
    final_df.to_csv(MASTER_CSV_PATH, index=False)
    print(f"Saved {len(final_df)} records to {MASTER_CSV_PATH}")
    
    # Append to History (Optimized: Check for Changes)
    new_history_rows = []
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    skipped_count = 0
    logged_count = 0
    
    # Pre-process history for fast lookup
    last_history_state = {}
    if not history_df.empty:
        # Sort by date
        history_df = history_df.sort_values('Date_Time')
        # Group by name and take last
        last_state_df = history_df.drop_duplicates(subset=['IPO_Name'], keep='last')
        for _, r in last_state_df.iterrows():
            last_history_state[r['IPO_Name']] = {
                'GMP': float(r['GMP']),
                'Sub': float(r['Sub'])
            }
            
    for _, row in merged_df.iterrows():
        name = row['Name']
        new_gmp = float(row['GMP'])
        new_sub = float(row['Sub_QIB'])
        
        # Only consider valid data
        if new_gmp == 0 and new_sub == 0:
            continue
            
        should_log = False
        
        if name in last_history_state:
            last = last_history_state[name]
            # Check for modification
            if new_gmp != last['GMP'] or new_sub != last['Sub']:
                should_log = True
        else:
            # New IPO or no history -> Log it
            should_log = True
            
        if should_log:
             new_history_rows.append({
                 'Date_Time': current_time,
                 'IPO_Name': name,
                 'GMP': new_gmp,
                 'Sub': new_sub, # Tracking QIB as main sub metric
                 'Nifty_Trend': live_nifty_trend, # Store live trend in history
                 'VIX': live_vix # Store live VIX in history
             })
             logged_count += 1
        else:
             skipped_count += 1
             
    if new_history_rows:
        new_history_df = pd.DataFrame(new_history_rows)
        # Append to existing history
        if os.path.exists(GMP_HISTORY_CSV_PATH):
            history_df = pd.concat([history_df, new_history_df], ignore_index=True)
        else:
            history_df = new_history_df
            
        history_df.to_csv(GMP_HISTORY_CSV_PATH, index=False)
        print(f"History Optimization: Skipped {skipped_count} duplicate rows. Logged {logged_count} new changes.")
        print(f"Appended {len(new_history_rows)} records to GMP History.")
    else:
        print(f"History Optimization: Skipped {skipped_count} duplicate rows. No new changes to log.")

    # 4. Trigger Training
    print("\n--- Triggering Training ---")
    exit_code = os.system("python3 src/train_model.py")
    if exit_code != 0:
        print("Training script failed.")
    else:
        print("Training completed.")

if __name__ == "__main__":
    main()
