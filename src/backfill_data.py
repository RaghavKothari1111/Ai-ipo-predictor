import requests
import pandas as pd
import os
import re
from datetime import datetime
from bs4 import BeautifulSoup

# Configuration
DATA_DIR = "data"
MASTER_CSV_PATH = os.path.join(DATA_DIR, "master_ipo_v3.csv")
CHITTORGARH_URL_TEMPLATE = "https://webnodejs.chittorgarh.com/cloud/report/data-read/22/{day}/{month}/{year}/{fy}/0/0/0?search=&v=08-39"

def get_financial_year(date_obj):
    if date_obj.month >= 4:
        start_year = date_obj.year
        end_year = (date_obj.year + 1) % 100
    else:
        start_year = date_obj.year - 1
        end_year = date_obj.year % 100
    return f"{start_year}-{end_year}"

def clean_html(raw_html):
    if not isinstance(raw_html, str):
        return str(raw_html)
    soup = BeautifulSoup(raw_html, "html.parser")
    return soup.get_text(strip=True)

def parse_name(raw_name):
    """Normalizes company name."""
    if not raw_name: return "Unknown"
    name_clean = clean_html(raw_name)
    # Remove "BSE SME", "NSE SME"
    normalized = re.sub(r'(?:BSE|NSE)\s*SME.*', '', name_clean).strip()
    # Remove "IPOL@..." suffix if present (from dirty master data)
    normalized = re.sub(r'\s*IPOL@.*', '', normalized).strip()
    return normalized

def fetch_chittorgarh_data():
    now = datetime.now()
    day = now.day
    month = now.month
    year = now.year
    fy = get_financial_year(now)
    
    url = CHITTORGARH_URL_TEMPLATE.format(day=day, month=month, year=year, fy=fy)
    print(f"Fetching from: {url}")
    
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        return data.get("reportTableData", [])
    except Exception as e:
        print(f"Error fetching data: {e}")
        return []

def process_chittorgarh(data):
    processed = []
    if data:
        print("First row keys:", data[0].keys())
        print("First row Name:", data[0].get("Name"))
        
    for row in data:
        raw_name = row.get("Company Name", "")
        name = parse_name(raw_name)
        
        # Issue Size
        try:
            size_str = clean_html(row.get("Total Issue Amount (Incl.Firm reservations) (Rs.cr.)", "")).replace(',', '')
            match = re.search(r'([\d\.]+)', size_str)
            issue_size_cr = float(match.group(1)) if match else 0.0
        except:
            issue_size_cr = 0.0
            
        # Subscriptions
        def parse_sub(key):
            try:
                val = clean_html(row.get(key, "")).replace('x', '').replace(',', '')
                return float(val)
            except:
                return 0.0
        
        sub_qib = parse_sub("QIB (x)")
        sub_nii = parse_sub("NII (x)")
        sub_retail = parse_sub("Retail (x)")
        
        processed.append({
            "MatchName": name, # Use this for matching
            "IPO_Size_Cr": issue_size_cr,
            "Sub_QIB": sub_qib,
            "Sub_NII": sub_nii,
            "Sub_Retail": sub_retail
        })
    return pd.DataFrame(processed)

def backfill():
    if not os.path.exists(MASTER_CSV_PATH):
        print("Master dataset not found.")
        return

    print(f"Loading {MASTER_CSV_PATH}...")
    master_df = pd.read_csv(MASTER_CSV_PATH)
    
    # Create a MatchName column for master_df to handle dirty names
    master_df['MatchName'] = master_df['Name'].apply(parse_name)
    
    print("Fetching Chittorgarh data...")
    raw_data = fetch_chittorgarh_data()
    fund_df = process_chittorgarh(raw_data)
    
    print(f"Fetched {len(fund_df)} records.")
    
    # Merge
    # We want to update master_df with columns from fund_df where MatchName matches
    # We use left join to keep all master rows
    merged = pd.merge(master_df, fund_df, on="MatchName", how="left", suffixes=('', '_new'))
    
    # Update columns
    cols_to_update = ['IPO_Size_Cr', 'Sub_QIB', 'Sub_NII', 'Sub_Retail']
    updated_count = 0
    
    for index, row in merged.iterrows():
        # Check if we have new data
        if pd.notnull(row.get('Sub_QIB_new')):
            # Update the original master_df at this index
            # Note: merged index might not match master_df index if merge changed order, 
            # but here we did left join on master, so order should be preserved? 
            # Actually, merge resets index usually.
            # Safer to iterate master_df and look up in fund_df dict.
            pass
            
    # Deduplicate fund_df by MatchName
    fund_df.drop_duplicates(subset=['MatchName'], keep='last', inplace=True)
    
    print("Sample Master MatchNames:", master_df['MatchName'].head(5).tolist())
    print("Sample Fund MatchNames:", fund_df['MatchName'].head(5).tolist())
    
    # Create a dictionary of Fund Data
    fund_dict = fund_df.set_index('MatchName').to_dict('index')
    fund_names = list(fund_dict.keys())
    
    import difflib
    
    for index, row in master_df.iterrows():
        match_name = row['MatchName']
        
        # Try exact match first
        if match_name in fund_dict:
            best_match = match_name
        else:
            # Try fuzzy match
            matches = difflib.get_close_matches(match_name, fund_names, n=1, cutoff=0.6)
            if matches:
                best_match = matches[0]
                # print(f"Fuzzy Match: '{match_name}' -> '{best_match}'")
            else:
                best_match = None
        
        if best_match:
            fund_data = fund_dict[best_match]
            
            # Update fields
            master_df.at[index, 'IPO_Size_Cr'] = fund_data['IPO_Size_Cr']
            master_df.at[index, 'Sub_QIB'] = fund_data['Sub_QIB']
            master_df.at[index, 'Sub_NII'] = fund_data['Sub_NII']
            master_df.at[index, 'Sub_Retail'] = fund_data['Sub_Retail']
            updated_count += 1
            
    # Drop MatchName
    master_df.drop(columns=['MatchName'], inplace=True)
    
    # Save
    master_df.to_csv(MASTER_CSV_PATH, index=False)
    print(f"Backfill complete. Updated {updated_count} records.")

if __name__ == "__main__":
    backfill()
