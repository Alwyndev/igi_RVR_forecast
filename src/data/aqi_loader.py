"""
aqi_loader.py — Loader for CPCB AQI Hourly Data

Loads PM2.5 and PM10 hourly values from the Excel file and upsamples
to 10-minute intervals using linear interpolation per user specification.
"""

import pandas as pd
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

def load_and_resample_aqi(excel_path: str) -> pd.DataFrame:
    """
    Loads AQI data, parses the datetime, and interpolates from 1-hour
    to 10-minute frequency.
    
    Args:
        excel_path: Path to PM2.5_PM10_Hourly_Delhi_2018-2025.xlsx
        
    Returns:
        DataFrame with 10-minute DatetimeIndex
    """
    logger.info(f"Loading AQI data from {excel_path}")
    
    # Read the Excel file. Assuming columns are like 'From Date', 'PM2.5', 'PM10'
    # We might need to inspect the exact column names later, but for now we write
    # generic logic based on standard CPCB formats.
    try:
        df = pd.read_excel(excel_path)
    except Exception as e:
        logger.error(f"Failed to read AQI Excel file: {e}")
        raise
        
    # Standardize column names
    col_map = {}
    for col in df.columns:
        if 'Date' in col or 'Time' in col:
            col_map[col] = 'datetime'
        elif 'PM2.5' in str(col).upper():
            col_map[col] = 'pm25'
        elif 'PM10' in str(col).upper():
            col_map[col] = 'pm10'
            
    df.rename(columns=col_map, inplace=True)
    
    if 'datetime' not in df.columns:
        # Fallback if the first column is the date
        df.rename(columns={df.columns[0]: 'datetime'}, inplace=True)
        
    # Convert to datetime and set as index
    df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')
    df.dropna(subset=['datetime'], inplace=True)
    df.set_index('datetime', inplace=True)
    df.sort_index(inplace=True)
    
    # Ensure numeric types, coerce 'None', 'NA', etc to NaN
    for col in ['pm25', 'pm10']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
    # Drop duplicates if any
    df = df[~df.index.duplicated(keep='last')]
    
    logger.info("Upsampling AQI from 1-hour to 10-minute via linear interpolation...")
    
    # Resample from 1H to 10min. 
    # asfreq() inserts NaN for the new 10-min bins
    resampled_df = df.resample('10min').asfreq()
    
    # Linear interpolate up to 6 gaps (1 hour)
    # This smoothly transitions pollution levels between hour marks
    resampled_df = resampled_df.interpolate(method='linear', limit=6)
    
    return resampled_df[['pm25', 'pm10']]
