"""
rvr_parser.py — Parser for IGIA RVR 10-second instrument data

Parses tab-delimited .txt files, handles missing values (**** -> NaN),
consolidates OLD/NEW sensors, and resamples to 10-minute intervals
using Mean and Min aggregation.
"""

import os
import glob
import pandas as pd
import numpy as np
import logging
from typing import List, Optional
from tqdm import tqdm
from pathlib import Path
from datetime import datetime

# Import spatial configuration
from .runway_config import consolidate_zone_name, RVR_FOLDER_TO_ZONE, RVR_2024_FOLDER_TO_ZONE

logger = logging.getLogger(__name__)

# The raw RVR columns based on our inspection
RVR_COLUMNS = [
    "time",
    "rvr_limited",
    "mor_limited",
    "rvr_actual",
    "mor_actual",
    "blm",
    "transmissivity",
    "ref_voltage",
    "pd_voltage"
]

def parse_single_rvr_file(filepath: str, date_str: str, zone_name: str) -> Optional[pd.DataFrame]:
    """
    Parses a single RVR .txt file into a pandas DataFrame.
    
    Args:
        filepath: Full path to the .txt file
        date_str: Date string in DD.MM.YYYY format
        zone_name: Canonical zone name (e.g., '09_TDZ')
        
    Returns:
        DataFrame with datetime index and numeric columns, or None if file empty/corrupt
    """
    try:
        # Check if file has data beyond the 8-line header
        if os.path.getsize(filepath) < 200:
            return None
            
        # Read the file skipping the 8-line header block
        # Using '\t' separator, but some files might use spaces if corrupted
        df = pd.read_csv(filepath, sep=r'\t+', engine='python', skiprows=8, names=RVR_COLUMNS, header=None)
        
        if df.empty:
            return None
            
        # Create full datetime from date_str and time column
        # Format of date_str: '01.01.2024', time: '00:00:18'
        datetime_strs = f"{date_str} " + df['time'].astype(str)
        # Fast parsing - any broken rows become NaT instantly
        df.index = pd.to_datetime(datetime_strs, format="%d.%m.%Y %H:%M:%S", errors='coerce')
        
        # Drop the now-redundant time column
        df = df.drop(columns=['time'])
        
        # Drop rows where datetime parsing failed (NaT)
        df = df[df.index.notna()]
        
        # Add the zone column (consolidated to OLD/canonical name)
        df['zone'] = consolidate_zone_name(zone_name)
        
        # Coerce all value columns to numeric. The errors='coerce' turns string 
        # error codes like '****' or plain spaces into NaN, which is our 
        # exact missing-value strategy.
        for col in df.columns:
            if col != 'zone':
                df[col] = pd.to_numeric(df[col], errors='coerce')
                
        return df

    except Exception as e:
        logger.warning(f"Error parsing {filepath}: {str(e)}")
        return None


def resample_rvr_to_10min(df: pd.DataFrame) -> pd.DataFrame:
    """
    Resamples 10-second RVR data to 10-minute intervals.
    Per user spec: uses both Mean and Min aggregation for RVR numeric columns.
    
    Args:
        df: Raw 10-second DataFrame with datetime index
        
    Returns:
        Resampled 10-minute DataFrame with multi-level columns flattened
    """
    if df.empty:
        return df
        
    # We group by the zone (even though it should be uniform per DF chunk, 
    # it's safe) and then resample the index to 10 minutes (10T or 10min).
    # We apply both mean and min aggregations.
    
    # Exclude 'zone' from numeric aggregation, but we keep it
    zone_val = df['zone'].iloc[0]
    numeric_df = df.drop(columns=['zone'])
    
    resampled = numeric_df.resample('10min').agg(['mean', 'min'])
    
    # Flatten the MultiIndex columns: e.g., ('rvr_actual', 'mean') -> 'rvr_actual_mean'
    resampled.columns = ['_'.join(col).strip() for col in resampled.columns.values]
    
    # Re-add the zone column
    resampled['zone'] = zone_val
    
    return resampled


def build_rvr_dataset(data_root: str) -> pd.DataFrame:
    """
    Crawls both RVR directories, parses all files, deduplicates, 
    and returns a single unified 10-minute resampled DataFrame.
    
    Args:
        data_root: Base path to IGI_Antigravity folder
    """
    logger.info("Starting RVR data ingestion and parsing...")
    
    rvr_2019_23_path = os.path.join(data_root, "Latest Data", "RVR")
    rvr_2024_25_path = os.path.join(data_root, "Latest Data", "RVR DATA 2024-25")
    
    all_chunks = []
    
    # Process 2019-2023 files
    if os.path.exists(rvr_2019_23_path):
        _process_directory(rvr_2019_23_path, RVR_FOLDER_TO_ZONE, all_chunks)
        
    # Process 2024-2025 files
    if os.path.exists(rvr_2024_25_path):
        _process_directory(rvr_2024_25_path, RVR_2024_FOLDER_TO_ZONE, all_chunks)
        
    if not all_chunks:
        raise ValueError("No RVR data found or parsed successfully.")
        
    logger.info("Concatenating all parsed chunks...")
    full_df = pd.concat(all_chunks)
    
    # The index is datetime, but there may be duplicates due to the 2024 overlap 
    # between the two folders. Since we processed the 2024-25 folder *second*, 
    # its rows are appended last. By keeping the 'last' duplicate, we enforce 
    # the rule: "Use Latest Data/RVR DATA 2024-25/ as canonical for 2024+".
    
    # To drop duplicates cleanly, we need 'zone' from the columns and the datetime index
    logger.info("Deduplicating overlapping timestamps...")
    
    # Temporary reset of index to use drop_duplicates on (index + zone)
    # Guarantee the index is named 'datetime' before resetting
    full_df.index.name = 'datetime'
    full_df = full_df.reset_index()
    
    # Deduplicate based on exact time and zone. Keep the last one encountered.
    curr_len = len(full_df)
    full_df = full_df.drop_duplicates(subset=['datetime', 'zone'], keep='last')
    logger.info(f"Dropped {curr_len - len(full_df)} duplicate rows.")
    
    # Restore datetime index
    full_df.set_index('datetime', inplace=True)
    full_df.sort_index(inplace=True)
    
    return full_df


from concurrent.futures import ProcessPoolExecutor, as_completed

from concurrent.futures import ProcessPoolExecutor, as_completed

def _parse_and_resample(filepath, date_str, zone_canonical):
    """Worker function for multiprocessing."""
    raw_df = parse_single_rvr_file(filepath, date_str, zone_canonical)
    if raw_df is not None:
        return resample_rvr_to_10min(raw_df)
    return None

def _process_directory(base_path: str, mapping_dict: dict, chunks_list: List[pd.DataFrame]):
    """Helper to crawl a directory struct and append resampled DFs to list."""
    folders = os.listdir(base_path)
    for folder in folders:
        folder_path = os.path.join(base_path, folder)
        if not os.path.isdir(folder_path):
            continue
            
        zone_canonical = mapping_dict.get(folder)
        if not zone_canonical:
            logger.warning(f"Folder '{folder}' not recognized in runway mappings. Skipping.")
            continue
            
        # Find all .txt files recursively
        txt_files = glob.glob(os.path.join(folder_path, "**", "*.txt"), recursive=True)
        if not txt_files:
            continue
            
        logger.info(f"Processing {len(txt_files)} files for zone '{folder}' -> '{zone_canonical}' (Sequential)")
        
        for filepath in tqdm(txt_files, desc=folder, leave=False):
            filename = os.path.basename(filepath)
            date_str = filename.replace('.txt', '')
            
            raw_df = parse_single_rvr_file(filepath, date_str, zone_canonical)
            if raw_df is not None:
                resampled_df = resample_rvr_to_10min(raw_df)
                if resampled_df is not None:
                    chunks_list.append(resampled_df)
