"""
API Health Check - Diagnose corrupted data in the IPO Predictor pipeline.

Usage: python3 src/api_health_check.py

Checks:
1. All 5 API endpoints for connectivity
2. Identifies IPOs with missing subscription data
3. Lists corrupted entries in master_ipo_v3.csv and explains why
"""

import requests
import pandas as pd
import os
import re
from datetime import datetime

DATA_DIR = "data"
MASTER_CSV_PATH = os.path.join(DATA_DIR, "master_ipo_v3.csv")

# API Templates
APIS = {
    "Investorgain 331 (Hype/GMP)": "https://webnodejs.investorgain.com/cloud/report/data-read/331/{day}/{month}/{year}/{fy}/0/all?search=&v=09-18",
    "Investorgain 333 (Subscriptions)": "https://webnodejs.investorgain.com/cloud/report/data-read/333/{day}/{month}/{year}/{fy}/0/all?search=&v=23-49",
    "Chittorgarh 22 (SME Fundamentals)": "https://webnodejs.chittorgarh.com/cloud/report/data-read/22/{day}/{month}/{year}/{fy}/0/0/0?search=&v=08-39",
    "Chittorgarh 21 (Mainboard)": "https://webnodejs.chittorgarh.com/cloud/report/data-read/21/{day}/{month}/{year}/{fy}/0/0/0?search=&v=11-25",
    "Chittorgarh 82 (Issue Prices)": "https://webnodejs.chittorgarh.com/cloud/report/data-read/82/{day}/{month}/{year}/{fy}/0/all/0?search=&v=23-59",
}

def get_financial_year(date_obj):
    if date_obj.month >= 4:
        start_year = date_obj.year
        end_year = (date_obj.year + 1) % 100
    else:
        start_year = date_obj.year - 1
        end_year = date_obj.year % 100
    return f"{start_year}-{end_year:02d}"

def check_apis():
    print("=" * 60)
    print("  API HEALTH CHECK")
    print("=" * 60)
    
    now = datetime.now()
    params = {
        "day": now.day,
        "month": now.month,
        "year": now.year,
        "fy": get_financial_year(now),
    }
    
    all_ok = True
    
    for name, template in APIS.items():
        url = template.format(**params)
        print(f"\n--- {name} ---")
        print(f"URL: {url}")
        
        try:
            resp = requests.get(url, timeout=15)
            print(f"HTTP Status: {resp.status_code}")
            
            if resp.status_code != 200:
                print(f"❌ FAIL: Non-200 status code")
                all_ok = False
                continue
            
            data = resp.json()
            records = data.get("reportTableData", [])
            print(f"Records: {len(records)}")
            
            if len(records) == 0:
                print(f"⚠️ WARNING: Zero records returned")
                all_ok = False
            else:
                print(f"✅ OK")
                # Show sample keys
                print(f"Sample keys: {list(records[0].keys())[:8]}...")
                
        except requests.exceptions.Timeout:
            print(f"❌ FAIL: Request timed out (15s)")
            all_ok = False
        except requests.exceptions.ConnectionError:
            print(f"❌ FAIL: Connection error")
            all_ok = False
        except Exception as e:
            print(f"❌ FAIL: {e}")
            all_ok = False
    
    return all_ok

def check_csv_corruption():
    print("\n" + "=" * 60)
    print("  CSV DATA QUALITY CHECK")
    print("=" * 60)
    
    if not os.path.exists(MASTER_CSV_PATH):
        print(f"❌ Master CSV not found at {MASTER_CSV_PATH}")
        return
    
    df = pd.read_csv(MASTER_CSV_PATH)
    
    total = len(df)
    listed = df[df['Status'] == 'Listed']
    upcoming = df[df['Status'] == 'Upcoming']
    corrupted = df[df['Data_Stage'] == 'Corrupted']
    
    print(f"\nTotal records: {total}")
    print(f"Listed: {len(listed)}")
    print(f"Upcoming: {len(upcoming)}")
    print(f"Corrupted: {len(corrupted)}")
    
    if len(corrupted) > 0:
        print(f"\n--- Corrupted Entries ---")
        for _, row in corrupted.iterrows():
            name = row['Name']
            qib = pd.to_numeric(row.get('Sub_QIB', 0), errors='coerce')
            nii = pd.to_numeric(row.get('Sub_NII', 0), errors='coerce')
            retail = pd.to_numeric(row.get('Sub_Retail', 0), errors='coerce')
            ipo_size = pd.to_numeric(row.get('IPO_Size_Cr', 0), errors='coerce')
            
            # Treat NaN as 0 (missing/corrupt)
            qib = 0.0 if pd.isna(qib) else float(qib)
            nii = 0.0 if pd.isna(nii) else float(nii)
            retail = 0.0 if pd.isna(retail) else float(retail)
            ipo_size = 0.0 if pd.isna(ipo_size) else float(ipo_size)
            
            reasons = []
            if qib == 0 and nii == 0 and retail == 0:
                reasons.append("All subscriptions = 0 (API failed)")
            elif qib == 0:
                reasons.append("QIB = 0 (SME IPO, no institutional category)")
            if ipo_size == 0:
                reasons.append("IPO_Size = 0 (size parsing failed)")
            
            reason_str = " | ".join(reasons) if reasons else "Unknown"
            print(f"  ⚠️ {name}: QIB={qib}, NII={nii}, Retail={retail}, Size={ipo_size} → {reason_str}")
    
    # Check for potential issues in Listed IPOs
    print(f"\n--- Listed IPOs with Zero Issue Price ---")
    zero_price_listed = listed[listed['Issue_Price'] == 0]
    if len(zero_price_listed) > 0:
        for _, row in zero_price_listed.iterrows():
            print(f"  ⚠️ {row['Name']}: Issue_Price=0 (price parsing failed)")
    else:
        print("  ✅ All Listed IPOs have valid Issue Prices")
    
    # Check IPO_Size_Cr
    print(f"\n--- Listed IPOs with Zero IPO Size ---")
    zero_size_listed = listed[listed['IPO_Size_Cr'] == 0]
    if len(zero_size_listed) > 0:
        for _, row in zero_size_listed.iterrows():
            print(f"  ⚠️ {row['Name']}: IPO_Size_Cr=0")
    else:
        print("  ✅ All Listed IPOs have valid IPO Size")
    
    # Check VIX
    print(f"\n--- Listed IPOs with Zero VIX ---")
    if 'Avg_VIX' in df.columns:
        zero_vix = listed[listed['Avg_VIX'] == 0]
        if len(zero_vix) > 0:
            print(f"  ⚠️ {len(zero_vix)} Listed IPOs have Avg_VIX=0")
        else:
            print("  ✅ All Listed IPOs have valid Avg_VIX")
    
    if 'Current_VIX' in listed.columns:
        zero_cvix = listed[listed['Current_VIX'] == 0]
        if len(zero_cvix) > 0:
            print(f"  ⚠️ {len(zero_cvix)} Listed IPOs have Current_VIX=0")
        else:
            print("  ✅ All Listed IPOs have valid Current_VIX")
    else:
        print("  ⚠️ Current_VIX column is missing from the dataset")

if __name__ == "__main__":
    apis_ok = check_apis()
    check_csv_corruption()
    
    print("\n" + "=" * 60)
    if apis_ok:
        print("  ✅ All APIs are responding. Data issues may be from parsing bugs.")
    else:
        print("  ⚠️ Some APIs are not responding. This will cause corrupted data.")
    print("=" * 60)
