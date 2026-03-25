"""
metar_parser.py — Parser for ASOS METAR strings

Extracts structured meteorological features (wind, vis, temp, pressure)
from raw METAR strings using the python-metar library and regex.
Upsamples to 10-minute intervals via linear interpolation.
"""

import pandas as pd
import numpy as np
import logging
from metar.Metar import Metar, ParserError

logger = logging.getLogger(__name__)

def parse_metar_string(metar_str: str) -> dict:
    """
    Parses a single METAR string and extracts key features.
    
    Args:
        metar_str: Raw string like 'VIDP 311830Z 06003KT 1200 BR ...'
        
    Returns:
        Dict of extracted numeric/categorical features
    """
    # Start with empty/NaN default values
    features = {
        'wind_dir': np.nan,
        'wind_speed_kt': np.nan,
        'visibility_m': np.nan,
        'temp_c': np.nan,
        'dewpoint_c': np.nan,
        'qnh_hpa': np.nan,
        'weather_str': '',
        'cloud_str': ''
    }
    
    if not isinstance(metar_str, str) or pd.isna(metar_str):
        return features
        
    try:
        obs = Metar(metar_str, strict=False)
        
        if obs.wind_dir:
            features['wind_dir'] = obs.wind_dir.value()
        if obs.wind_speed:
            features['wind_speed_kt'] = obs.wind_speed.value("KT")
            
        if obs.vis:
            features['visibility_m'] = obs.vis.value("M")
            
        if obs.temp:
            features['temp_c'] = obs.temp.value("C")
        if obs.dewpt:
            features['dewpoint_c'] = obs.dewpt.value("C")
            
        if obs.press:
            features['qnh_hpa'] = obs.press.value("MB")
            
        # Concatenate weather phenomena (e.g., 'BR', 'FG', 'HZ')
        if obs.weather:
            codes = []
            for wx in obs.weather:
                # wx is a list of tuples like (intensity, description, precipitation, obscuration, other)
                # Filter out None values and join with spaces
                wx_str = "".join([c for c in wx if c])
                codes.append(wx_str)
            features['weather_str'] = " ".join(codes)
            
        # Cloud codes (e.g., 'NSC', 'SCT020')
        if obs.sky:
            codes = []
            for cloud in obs.sky:
                # cloud is a tuple like (coverage, height, characteristic)
                sky_str = "".join([str(c.value()) if hasattr(c, 'value') else str(c) for c in cloud if c])
                codes.append(sky_str)
            features['cloud_str'] = " ".join(codes)
            
    except ParserError as e:
        # Some badly mangled METARs might fail parsing; we gracefully return NaNs
        pass
    except Exception as e:
        pass
        
    return features


def process_asos_data(asos_csv_path: str) -> pd.DataFrame:
    """
    Loads ASOS CSV data, parses all METAR strings, and upsamples from 
    30-minute to 10-minute intervals using linear interpolation.
    
    Args:
        asos_csv_path: Path to the ASOS CSV file (e.g., all_data.csv)
        
    Returns:
        DataFrame with 10-minute DatetimeIndex
    """
    logger.info(f"Loading ASOS data from {asos_csv_path}")
    df = pd.read_csv(asos_csv_path)
    
    # Ensure datetime index
    df['valid'] = pd.to_datetime(df['valid'])
    df.set_index('valid', inplace=True)
    df.sort_index(inplace=True)
    
    # Drop duplicates if any (keep last)
    df = df[~df.index.duplicated(keep='last')]
    
    logger.info(f"Parsing {len(df)} METAR strings...")
    
    # Apply parser
    # We use apply(pd.Series) to unpack the returned dict into columns
    parsed_features = df['metar'].apply(parse_metar_string).apply(pd.Series)
    
    # Merge back to retain datetime index
    df_merged = pd.concat([df[['station']], parsed_features], axis=1)
    
    logger.info("Upsampling ASOS features from ~30-min to strictly 10-min via linear interpolation...")
    
    # Resample to 10-min
    # Since we want to upsample AND interpolate, we do it in steps.
    resampled_df = df_merged.resample('10min').asfreq()
    
    # Linear interpolation for continuous numeric vars
    numeric_cols = ['wind_dir', 'wind_speed_kt', 'visibility_m', 'temp_c', 'dewpoint_c', 'qnh_hpa']
    resampled_df[numeric_cols] = resampled_df[numeric_cols].interpolate(method='linear', limit=12) # max 2-hour gap limit
    
    # Forward-fill categoricals
    cat_cols = ['station', 'weather_str', 'cloud_str']
    resampled_df[cat_cols] = resampled_df[cat_cols].ffill(limit=12)
    
    return resampled_df
