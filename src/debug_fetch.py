
import requests
import json
from datetime import datetime

CHITTORGARH_SME_URL_TEMPLATE = "https://webnodejs.chittorgarh.com/cloud/report/data-read/22/{day}/{month}/{year}/{fy}/0/0/0?search=&v=08-39"
CHITTORGARH_MAIN_URL_TEMPLATE = "https://webnodejs.chittorgarh.com/cloud/report/data-read/21/{day}/{month}/{year}/{fy}/0/0/0?search=&v=11-25"

def get_financial_year(date_obj):
    if date_obj.month >= 4:
        start_year = date_obj.year
        end_year = (date_obj.year + 1) % 100
    else:
        start_year = date_obj.year - 1
        end_year = date_obj.year % 100
    return f"{start_year}-{end_year}"

def fetch_and_inspect():
    now = datetime.now()
    day = now.day
    month = now.month
    year = now.year
    fy = get_financial_year(now)
    
    url_sme = CHITTORGARH_SME_URL_TEMPLATE.format(day=day, month=month, year=year, fy=fy)
    url_main = CHITTORGARH_MAIN_URL_TEMPLATE.format(day=day, month=month, year=year, fy=fy)
    
    for url in [url_sme, url_main]:
        print(f"Fetching: {url}")
        try:
            resp = requests.get(url)
            data = resp.json().get("reportTableData", [])
            print(f"Fetched {len(data)} records.")
            
            for row in data:
                name = row.get("Company Name", "")
                if "ksh" in name.lower() or "pytochem" in name.lower():
                    print(f"\n--- FOUND: {name} ---")
                    # Print keys relevant to subscription
                    for k, v in row.items():
                        # print all columns to be sure
                         print(f"{k}: {v}")
                    
        except Exception as e:
            print(e)
        
fetch_and_inspect()
