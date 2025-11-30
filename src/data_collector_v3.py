import requests
import pandas as pd
import os
import re
import yfinance as yf
from datetime import datetime, timedelta
from bs4 import BeautifulSoup

# Configuration
# Base URL template. Categories: 'sme' or 'ipo' (Mainboard)
BASE_API_URL = "https://webnodejs.investorgain.com/cloud/report/data-read/331/{page}/11/2025/2025-26/0/{category}?search=&v=23-18"
DATA_DIR = "data"
MASTER_CSV_PATH = os.path.join(DATA_DIR, "master_ipo_v3.csv")

def fetch_data(category, page):
    """Fetches data from the API for a specific category and page."""
    url = BASE_API_URL.format(category=category, page=page)
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        return data.get("reportTableData", [])
    except Exception as e:
        print(f"Error fetching data for {category} page {page}: {e}")
        return []

# ... (keep other helper functions like clean_html, parse_name, etc. as they are) ...
# I will use multi_replace to target specific blocks if I can't replace the whole file easily, 
# but here I am replacing the top part and the main loop.
# Wait, replace_file_content replaces a contiguous block.
# I need to replace API_URL definition and fetch_data function, AND collect_and_save function.
# They are separated by other functions.
# I should use multi_replace_file_content or just overwrite the whole file again to be safe and clean, 
# as I have the full content in context.
# Overwriting is safer to ensure structure is correct.

# Let's overwrite the whole file with the new logic.


def clean_html(raw_html):
    """Removes HTML tags and decodes entities."""
    if not isinstance(raw_html, str):
        return str(raw_html)
    soup = BeautifulSoup(raw_html, "html.parser")
    return soup.get_text(strip=True)

def parse_name(raw_name):
    """
    Normalizes company name.
    Removes 'BSE SME', 'NSE SME', and suffixes like 'U', 'C', 'O'.
    Regex: re.sub(r'(?:BSE|NSE)\s*SME.*', '', raw_name).strip()
    """
    if not raw_name: return "Unknown"
    name_clean = clean_html(raw_name)
    # Remove BSE/NSE SME and everything after it (including status flags like U, C, O, L@...)
    # Wait, if we remove L@..., we might lose the listing price info if we don't extract it first.
    # But process_row extracts Listing Price from the raw name *before* this normalization if needed.
    # Actually, the user requirement says: "name = re.sub(r'(?:BSE|NSE)\s*SME.*', '', raw_name).strip()"
    # This will indeed strip L@... if it comes after BSE SME.
    # So we must ensure Listing Price is extracted from the RAW name before this function is called or inside process_row.
    
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

def parse_gmp_low_high(val):
    """Splits 0 ↓ / 0 ↑ into (0.0, 0.0)"""
    if not val: return 0.0, 0.0
    val = clean_html(val)
    parts = val.split('/')
    if len(parts) == 2:
        try:
            low = float(re.sub(r'[^\d\.]', '', parts[0]))
            high = float(re.sub(r'[^\d\.]', '', parts[1]))
            return low, high
        except:
            pass
    return 0.0, 0.0

def parse_anchor(val):
    """
    Anchor Logic:
    Checkmark/True/Yes -> 1
    Cross/False/No/Empty -> 0
    """
    if not val: return 0
    val_str = str(val).lower()
    if '✅' in val_str or 'true' in val_str or 'yes' in val_str:
        return 1
    return 0

def parse_date(date_str):
    """
    Parses date like '12-Dec' to 'YYYY-MM-DD'.
    """
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

def get_market_mood(target_date_str):
    """
    Fetches Nifty 50 and India VIX data.
    Fixes:
    1. If target_date > Today -> Use Today.
    2. Weekend/Holiday -> Look back up to 3 days.
    3. Safety Net -> Default VIX 12.0.
    """
    try:
        if not target_date_str:
            target_date = datetime.now()
        else:
            try:
                target_date = datetime.strptime(target_date_str, "%Y-%m-%d")
            except:
                target_date = datetime.now()

        # Fix 1: Future Check
        if target_date > datetime.now():
            target_date = datetime.now()

        # Fix 2: Lookback Logic (handled by fetching a range ending at target_date)
        # We want the CLOSEST available data ON or BEFORE target_date.
        # So we fetch [target_date - 5 days, target_date + 1 day]
        
        start_date = target_date - timedelta(days=5)
        end_date = target_date + timedelta(days=1)
        
        # Suppress yfinance progress
        nifty = yf.download("^NSEI", start=start_date, end=end_date, progress=False)
        vix = yf.download("^INDIAVIX", start=start_date, end=end_date, progress=False)
        
        # Fix 3: Safety Net
        if vix.empty:
            vix_val = 12.0
        else:
            # Get last available row (closest to target_date)
            vix_val = float(vix['Close'].iloc[-1])
            if pd.isna(vix_val) or vix_val == 0.0:
                vix_val = 12.0

        if nifty.empty:
            nifty_trend = 0.0
        else:
            nifty_close = float(nifty['Close'].iloc[-1])
            
            # Trend 7D calculation
            # We need price 7 days before the *effective* target date
            effective_date = nifty.index[-1]
            date_7d_ago = effective_date - timedelta(days=7)
            
            # Fetch history for trend if needed, or just fetch a longer range initially?
            # Let's fetch a single point for 7d ago to be efficient/simple
            # Or better, just fetch 15 days history initially.
            # Let's re-fetch if we don't have enough history in the dataframe?
            # Actually, let's just fetch 15 days in the first place.
            
            # Re-do fetch with longer window
            start_date_long = target_date - timedelta(days=15)
            nifty_long = yf.download("^NSEI", start=start_date_long, end=end_date, progress=False)
            
            if nifty_long.empty:
                nifty_trend = 0.0
            else:
                current_price = float(nifty_long['Close'].iloc[-1])
                # Find index 7 days ago
                try:
                    # closest index to date_7d_ago
                    idx = nifty_long.index.get_indexer([date_7d_ago], method='nearest')[0]
                    price_7d_ago = float(nifty_long['Close'].iloc[idx])
                    nifty_trend = (current_price - price_7d_ago) / price_7d_ago
                except:
                    nifty_trend = 0.0

        return nifty_trend, vix_val

    except Exception as e:
        print(f"Market Data Error: {e}")
        return 0.0, 12.0 # Default VIX 12

def process_row(row):
    # 1. Basic Extraction
    raw_name = row.get("Name", "")
    raw_anchor = row.get("Anchor", "")
    raw_listing = row.get("~Str_Listing", "") or row.get("Listing", "")
    
    # 2. Extract Listing Price BEFORE Name Normalization
    name_clean_for_extraction = clean_html(raw_name)
    match = re.search(r'L@([\d\.]+)', name_clean_for_extraction)
    listing_price = float(match.group(1)) if match else None
    
    # 3. Name Normalization (Fix 1)
    name = parse_name(raw_name)
    
    # 4. Anchor
    has_anchor = parse_anchor(raw_anchor)
    
    # 5. GMP
    gmp = parse_currency(row.get("GMP", ""))
    gmp_low, gmp_high = parse_gmp_low_high(row.get("GMP(L/H)", ""))
    
    # 6. Dates
    listing_date = parse_date(raw_listing)
    
    # 7. Market Mood
    if listing_date:
        target_date = listing_date
    else:
        target_date = datetime.now().strftime("%Y-%m-%d")
        
    nifty_trend, vix = get_market_mood(target_date)
    
    # 8. Other Fields
    try:
        sub = float(clean_html(row.get("Sub", "")).replace('x', '').replace(',', ''))
    except:
        sub = 0.0
        
    try:
        ipo_size = float(clean_html(row.get("IPO Size", "")).replace(',', ''))
    except:
        ipo_size = 0.0

    return {
        "Name": name,
        "Has_Anchor": has_anchor,
        "GMP": gmp,
        "GMP_High": gmp_high,
        "Sub": sub,
        "IPO_Size": ipo_size,
        "Listing_Date": listing_date,
        "Listing_Price": listing_price,
        "Nifty_Trend_7D": nifty_trend,
        "India_VIX_Close": vix
    }

def collect_and_save():
    all_data = []
    categories = ['sme', 'ipo'] # SME and Mainboard
    
    for category in categories:
        page = 1
        print(f"Fetching {category.upper()} data...")
        while True:
            print(f"  Fetching page {page}...", end='\r')
            page_data = fetch_data(category, page)
            
            if not page_data:
                print(f"\n  No more data for {category} at page {page}. Moving to next category.")
                break
                
            all_data.extend(page_data)
            page += 1
            
    print(f"\nTotal records fetched: {len(all_data)}")
    
    processed_rows = []
    for i, row in enumerate(all_data):
        print(f"Processing {i+1}/{len(all_data)}...", end='\r')
        processed_rows.append(process_row(row))
    print("\nProcessing complete.")
        
    new_df = pd.DataFrame(processed_rows)
    
    os.makedirs(DATA_DIR, exist_ok=True)
    
    # Self-Healing Merge Logic
    if os.path.exists(MASTER_CSV_PATH):
        try:
            print("Loading existing dataset for self-healing...")
            master_df = pd.read_csv(MASTER_CSV_PATH)
            
            # 1. Clean Old Data (Regex Transformation)
            master_df['Name'] = master_df['Name'].astype(str).apply(
                lambda x: re.sub(r'(?:BSE|NSE)\s*SME.*', '', x).strip()
            )
            
            # 2. Deduplicate Old Data
            before_dedup = len(master_df)
            master_df = master_df.drop_duplicates(subset=['Name'], keep='last')
            print(f"Cleaned and deduplicated existing data. Rows: {before_dedup} -> {len(master_df)}")
            
            # 3. Create Dictionary for Upsert
            master_dict = {row['Name']: row for _, row in master_df.iterrows()}
            
            # 4. Upsert New Data
            for _, row in new_df.iterrows():
                master_dict[row['Name']] = row
                
            final_df = pd.DataFrame(list(master_dict.values()))
            
        except Exception as e:
            print(f"Error reading/processing master csv: {e}. Overwriting with new data.")
            final_df = new_df
    else:
        final_df = new_df
        
    final_df.to_csv(MASTER_CSV_PATH, index=False)
    print(f"Saved {len(final_df)} records to {MASTER_CSV_PATH}")
    
    # Trigger Training
    print("Triggering Training...")
    try:
        import sys
        sys.path.append(os.getcwd())
        from src.train_v3 import train_model
        train_model()
    except Exception as e:
        print(f"Could not import train_model: {e}")
        os.system("python3 src/train_v3.py")

if __name__ == "__main__":
    collect_and_save()
