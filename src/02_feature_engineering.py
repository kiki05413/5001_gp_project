"""
DSC5001 Project: Hong Kong AQHI High-Risk Hour Early Warning
=============================================================
Step 2: Data processing, feature engineering, and target construction

- Parse AQHI hourly CSV files (2021-2024)
- Merge with HKO daily weather data
- Construct lagged features, time encodings
- Create binary target: next-hour AQHI >= 7
"""

import os
import glob
import re
import numpy as np
import pandas as pd
from datetime import datetime

RAW_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'raw_data'))
DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data'))
os.makedirs(DATA_DIR, exist_ok=True)

# Station type mapping
ROADSIDE_STATIONS = ['Causeway Bay', 'Central', 'Mong Kok']
GENERAL_STATIONS = [
    'Central/Western', 'Southern', 'Eastern', 'Kwun Tong', 'Sham Shui Po',
    'Kwai Chung', 'Tsuen Wan', 'Tseung Kwan O', 'Yuen Long', 'Tuen Mun',
    'Tung Chung', 'Tai Po', 'Sha Tin', 'North', 'Tap Mun'
]

# AQHI risk threshold
AQHI_HIGH_RISK_THRESHOLD = 7


def parse_aqhi_file(filepath):
    """Parse a single AQHI monthly CSV file."""
    with open(filepath, 'r', encoding='utf-8-sig') as f:
        lines = f.readlines()
    
    # Find header line (starts with "Date,Hour")
    header_idx = None
    for i, line in enumerate(lines):
        if line.strip().startswith('Date,Hour'):
            header_idx = i
            break
    
    if header_idx is None:
        return pd.DataFrame()
    
    # Read the data
    headers = lines[header_idx].strip().split(',')
    data_lines = lines[header_idx + 1:]
    
    records = []
    current_date = None
    
    for line in data_lines:
        line = line.strip()
        if not line:
            continue
        parts = line.split(',')
        
        # Date might be empty (continuation from previous row)
        if parts[0].strip():
            current_date = parts[0].strip()
        
        if current_date is None:
            continue
            
        hour_str = parts[1].strip() if len(parts) > 1 else ''
        if not hour_str:
            continue
        
        try:
            hour = int(hour_str)
        except ValueError:
            continue
        
        # Parse station values
        record = {'Date': current_date, 'Hour': hour}
        for j, station in enumerate(headers[2:]):
            if j + 2 < len(parts):
                val = parts[j + 2].strip()
                # Remove asterisks (data from similar station)
                val = val.replace('*', '')
                try:
                    record[station] = int(val) if val else np.nan
                except ValueError:
                    record[station] = np.nan
            else:
                record[station] = np.nan
        
        records.append(record)
    
    df = pd.DataFrame(records)
    return df


def load_all_aqhi_data():
    """Load and combine all AQHI monthly files."""
    print("Loading AQHI data...")
    
    all_dfs = []
    aqhi_files = sorted(glob.glob(os.path.join(RAW_DIR, '*_Eng.csv')))
    
    # Filter to 2021-2024
    aqhi_files = [f for f in aqhi_files if re.match(r'.*202[1-4]\d{2}_Eng\.csv$', f)]
    
    print(f"  Found {len(aqhi_files)} monthly files")
    
    for filepath in aqhi_files:
        df = parse_aqhi_file(filepath)
        if not df.empty:
            all_dfs.append(df)
    
    combined = pd.concat(all_dfs, ignore_index=True)
    
    # Create datetime column
    combined['Date'] = pd.to_datetime(combined['Date'], format='%Y-%m-%d')
    combined['datetime'] = combined['Date'] + pd.to_timedelta(combined['Hour'], unit='h')
    
    # Sort
    combined = combined.sort_values('datetime').reset_index(drop=True)
    
    print(f"  Combined shape: {combined.shape}")
    print(f"  Date range: {combined['datetime'].min()} to {combined['datetime'].max()}")
    
    return combined


def reshape_to_long_format(df):
    """Reshape from wide (stations as columns) to long format."""
    print("Reshaping to long format...")
    
    station_columns = ROADSIDE_STATIONS + GENERAL_STATIONS
    available_stations = [s for s in station_columns if s in df.columns]
    
    long_df = df.melt(
        id_vars=['datetime', 'Date', 'Hour'],
        value_vars=available_stations,
        var_name='Station',
        value_name='AQHI'
    )
    
    # Drop rows with missing AQHI
    long_df = long_df.dropna(subset=['AQHI'])
    long_df['AQHI'] = long_df['AQHI'].astype(int)
    
    # Add station type
    long_df['StationType'] = long_df['Station'].apply(
        lambda x: 'Roadside' if x in ROADSIDE_STATIONS else 'General'
    )
    
    print(f"  Long format shape: {long_df.shape}")
    print(f"  Stations: {long_df['Station'].nunique()}")
    print(f"  Station types: {long_df['StationType'].value_counts().to_dict()}")
    
    return long_df


def load_weather_data():
    """Load and merge HKO daily weather data."""
    print("\nLoading weather data...")
    
    weather_dfs = {}
    
    def parse_hko_csv(filepath, param_name):
        """Parse HKO daily climate CSV, handling footer rows."""
        df = pd.read_csv(filepath, skiprows=2, encoding='utf-8-sig')
        # Keep only rows where Year column is numeric
        df = df[pd.to_numeric(df.iloc[:, 0], errors='coerce').notna()].copy()
        df.columns = ['Year', 'Month', 'Day', 'Value', 'Completeness']
        df['Year'] = df['Year'].astype(int)
        df['Month'] = df['Month'].astype(int)
        df['Day'] = df['Day'].astype(int)
        df['Value'] = pd.to_numeric(df['Value'].replace('***', np.nan), errors='coerce')
        # Build date string manually to avoid 1900-02-29 issues
        df = df[(df['Year'] >= 1884) & (df['Month'] >= 1) & (df['Month'] <= 12) & (df['Day'] >= 1) & (df['Day'] <= 31)]
        df['Date'] = pd.to_datetime(df['Year'].astype(str) + '-' + df['Month'].astype(str).str.zfill(2) + '-' + df['Day'].astype(str).str.zfill(2), errors='coerce')
        df = df.dropna(subset=['Date'])
        result = df[['Date', 'Value']].dropna()
        result.columns = ['Date', param_name]
        return result
    
    # Temperature
    temp_file = os.path.join(RAW_DIR, 'daily_HKO_TEMP_ALL.csv')
    if os.path.exists(temp_file):
        weather_dfs['Temp'] = parse_hko_csv(temp_file, 'Temp')
        print(f"  Temperature: {len(weather_dfs['Temp'])} records")
    
    # Relative Humidity
    rh_file = os.path.join(RAW_DIR, 'daily_HKO_RH_ALL.csv')
    if os.path.exists(rh_file):
        weather_dfs['RH'] = parse_hko_csv(rh_file, 'RH')
        print(f"  Humidity: {len(weather_dfs['RH'])} records")
    
    # Rainfall
    rf_file = os.path.join(RAW_DIR, 'daily_HKO_RF_ALL.csv')
    if os.path.exists(rf_file):
        weather_dfs['Rainfall'] = parse_hko_csv(rf_file, 'Rainfall')
        print(f"  Rainfall: {len(weather_dfs['Rainfall'])} records")
    
    # Wet Bulb Temperature
    wet_file = os.path.join(RAW_DIR, 'daily_HKO_WET_ALL.csv')
    if os.path.exists(wet_file):
        weather_dfs['WetBulb'] = parse_hko_csv(wet_file, 'WetBulb')
        print(f"  Wet Bulb: {len(weather_dfs['WetBulb'])} records")
    
    # Merge all weather data
    weather = None
    for param, df in weather_dfs.items():
        if weather is None:
            weather = df
        else:
            weather = weather.merge(df, on='Date', how='outer')
    
    print(f"  Merged weather shape: {weather.shape}")
    return weather


def engineer_features(df, weather):
    """Create all features for the classification task."""
    print("\nEngineering features...")
    
    # Ensure datetime is sorted per station
    df = df.sort_values(['Station', 'datetime']).reset_index(drop=True)
    
    # --- Time features ---
    df['hour'] = df['datetime'].dt.hour
    df['dayofweek'] = df['datetime'].dt.dayofweek
    df['month'] = df['datetime'].dt.month
    df['dayofyear'] = df['datetime'].dt.dayofyear
    
    # Cyclical encoding of time features
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['dow_sin'] = np.sin(2 * np.pi * df['dayofweek'] / 7)
    df['dow_cos'] = np.cos(2 * np.pi * df['dayofweek'] / 7)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    
    # Is weekend
    df['is_weekend'] = (df['dayofweek'] >= 5).astype(int)
    
    # --- Lagged AQHI features ---
    for lag in [1, 2, 3, 6, 12, 24]:
        df[f'AQHI_lag_{lag}'] = df.groupby('Station')['AQHI'].shift(lag)
    
    # --- Rolling statistics ---
    df['AQHI_rolling_mean_3'] = df.groupby('Station')['AQHI'].transform(
        lambda x: x.shift(1).rolling(3, min_periods=1).mean()
    )
    df['AQHI_rolling_mean_6'] = df.groupby('Station')['AQHI'].transform(
        lambda x: x.shift(1).rolling(6, min_periods=1).mean()
    )
    df['AQHI_rolling_std_6'] = df.groupby('Station')['AQHI'].transform(
        lambda x: x.shift(1).rolling(6, min_periods=1).std()
    )
    
    # AQHI trend (difference)
    df['AQHI_diff_1'] = df['AQHI_lag_1'] - df['AQHI_lag_2']
    df['AQHI_diff_3'] = df['AQHI_lag_1'] - df['AQHI_lag_3']
    
    # --- Target variable: next-hour AQHI >= 7 ---
    df['next_AQHI'] = df.groupby('Station')['AQHI'].shift(-1)
    df['target'] = (df['next_AQHI'] >= AQHI_HIGH_RISK_THRESHOLD).astype(int)
    
    # --- Merge weather data (by date) ---
    df['Date'] = pd.to_datetime(df['datetime'].dt.date)
    df = df.merge(weather, on='Date', how='left', suffixes=('', '_weather'))
    
    # Drop rows with missing target (last hour per station) or missing critical features
    critical_cols = ['AQHI_lag_1', 'AQHI_lag_2', 'AQHI_lag_3', 'next_AQHI', 'Temp', 'RH']
    df = df.dropna(subset=critical_cols)
    
    # Fill any remaining NaN in rolling features
    df['AQHI_rolling_std_6'] = df['AQHI_rolling_std_6'].fillna(0)
    
    # Rainfall might have NaN - fill with 0
    if 'Rainfall' in df.columns:
        df['Rainfall'] = df['Rainfall'].fillna(0)
    if 'WetBulb' in df.columns:
        df['WetBulb'] = df['WetBulb'].fillna(df['WetBulb'].median())
    
    # Station type encoding
    df['is_roadside'] = (df['StationType'] == 'Roadside').astype(int)
    
    print(f"  Final dataset shape: {df.shape}")
    print(f"  Target distribution:\n{df['target'].value_counts()}")
    print(f"  High-risk rate: {df['target'].mean():.4f} ({df['target'].mean()*100:.2f}%)")
    
    return df


def main():
    # Step 1: Load AQHI data
    aqhi_wide = load_all_aqhi_data()
    
    # Step 2: Reshape to long format
    aqhi_long = reshape_to_long_format(aqhi_wide)
    
    # Step 3: Load weather data
    weather = load_weather_data()
    
    # Step 4: Feature engineering
    data = engineer_features(aqhi_long, weather)
    
    # Save processed data
    output_path = os.path.join(DATA_DIR, 'processed_data.csv')
    data.to_csv(output_path, index=False)
    print(f"\nProcessed data saved to: {output_path}")
    print(f"Shape: {data.shape}")
    
    # Print summary statistics
    print("\n" + "=" * 60)
    print("DATASET SUMMARY")
    print("=" * 60)
    print(f"Total samples: {len(data):,}")
    print(f"Stations: {data['Station'].nunique()}")
    print(f"  Roadside: {data[data['is_roadside']==1]['Station'].nunique()}")
    print(f"  General: {data[data['is_roadside']==0]['Station'].nunique()}")
    print(f"Date range: {data['datetime'].min()} to {data['datetime'].max()}")
    print(f"High-risk (AQHI>=7) rate: {data['target'].mean()*100:.2f}%")
    print(f"\nAQHI statistics:")
    print(data['AQHI'].describe())
    print(f"\nFeature columns:")
    feature_cols = [c for c in data.columns if c not in 
                    ['datetime', 'Date', 'Hour', 'Station', 'StationType', 'AQHI', 'next_AQHI', 'target']]
    print(f"  {len(feature_cols)} features: {feature_cols}")
    
    return data


if __name__ == '__main__':
    main()
