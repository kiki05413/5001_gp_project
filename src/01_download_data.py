"""
DSC5001 Project: Hong Kong AQHI High-Risk Hour Early Warning
=============================================================
Step 1: Download AQHI hourly data and HKO daily weather data

Data sources:
- AQHI: https://www.aqhi.gov.hk/epd/ddata/html/history/YYYY/YYYYMM_Eng.csv
- Weather: https://data.weather.gov.hk/weatherAPI/cis/csvfile/HKO/ALL/daily_HKO_{PARAM}_ALL.csv
"""

import urllib.request
import os
import time
import sys

RAW_DIR = os.path.join(os.path.dirname(__file__), '..', 'raw_data')
os.makedirs(RAW_DIR, exist_ok=True)

# AQHI data: download 2021-2024 (4 years, avoiding 2025 AQHI methodology change)
AQHI_YEARS = [2021, 2022, 2023, 2024]
AQHI_MONTHS = list(range(1, 13))

# Weather data: HKO daily climate parameters
WEATHER_PARAMS = {
    'TEMP': 'daily_HKO_TEMP_ALL.csv',      # Daily Mean Temperature
    'RH':   'daily_HKO_RH_ALL.csv',        # Daily Mean Relative Humidity
    'RF':   'daily_HKO_RF_ALL.csv',        # Daily Total Rainfall
    'WET':  'daily_HKO_WET_ALL.csv',       # Daily Mean Wet Bulb Temperature
}

HEADERS = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}

def download_file(url, filepath, retries=3):
    """Download a file with retry logic."""
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers=HEADERS)
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = resp.read()
            with open(filepath, 'wb') as f:
                f.write(data)
            return True
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(2)
            else:
                print(f"  FAILED: {url} - {e}")
                return False
    return False


def download_aqhi_data():
    """Download monthly AQHI CSV files."""
    print("=" * 60)
    print("Downloading AQHI hourly data (2021-2024)...")
    print("=" * 60)
    success_count = 0
    fail_count = 0
    
    for year in AQHI_YEARS:
        for month in AQHI_MONTHS:
            filename = f"{year}{month:02d}_Eng.csv"
            filepath = os.path.join(RAW_DIR, filename)
            
            if os.path.exists(filepath) and os.path.getsize(filepath) > 100:
                success_count += 1
                continue
            
            url = f"https://www.aqhi.gov.hk/epd/ddata/html/history/{year}/{filename}"
            print(f"  Downloading {filename}...", end=" ")
            sys.stdout.flush()
            
            if download_file(url, filepath):
                print("OK")
                success_count += 1
            else:
                fail_count += 1
            
            time.sleep(0.5)  # Be polite to the server
    
    print(f"\nAQHI download complete: {success_count} success, {fail_count} failed")
    return success_count, fail_count


def download_weather_data():
    """Download HKO daily climate data."""
    print("\n" + "=" * 60)
    print("Downloading HKO daily weather data...")
    print("=" * 60)
    success_count = 0
    
    for param, filename in WEATHER_PARAMS.items():
        filepath = os.path.join(RAW_DIR, filename)
        
        if os.path.exists(filepath) and os.path.getsize(filepath) > 100:
            print(f"  {filename} - already exists")
            success_count += 1
            continue
        
        url = f"https://data.weather.gov.hk/weatherAPI/cis/csvfile/HKO/ALL/{filename}"
        print(f"  Downloading {filename}...", end=" ")
        sys.stdout.flush()
        
        if download_file(url, filepath):
            print("OK")
            success_count += 1
        else:
            print("FAILED")
    
    print(f"\nWeather download complete: {success_count}/{len(WEATHER_PARAMS)} success")
    return success_count


if __name__ == '__main__':
    aqhi_ok, aqhi_fail = download_aqhi_data()
    weather_ok = download_weather_data()
    
    print("\n" + "=" * 60)
    print("Data download summary:")
    print(f"  AQHI files: {aqhi_ok} downloaded, {aqhi_fail} failed")
    print(f"  Weather files: {weather_ok} downloaded")
    print(f"  Raw data directory: {os.path.abspath(RAW_DIR)}")
    print("=" * 60)
