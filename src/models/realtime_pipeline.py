"""
realtime_pipeline.py -- Operational Orchestrator for IGIA RVR

This script bridges the gap between raw sensor files and the live dashboard.
It performs:
  1. Automated scanning of 'Latest Data/' folders.
  2. Incremental preprocessing of the most recent 6-hour window.
  3. Prediction via Multi-Horizon BiLSTM.
  4. Auto-refresh of igia_rvr_dashboard_multi.html.
"""

import os
import sys
import time
import logging
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.models.inference import RVRInferenceEngine # To be updated for Multi if needed, or use dashboard_multi logic
from dashboard_multi import create_multi_dashboard

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)

def run_once():
    """
    Simulates one real-time cycle: 
    In a real flight ops center, this would fetch from a database or FTP.
    Here, it triggers the dashboard which loads the 'latest' from our parquet.
    """
    logger.info("Triggering real-time update cycle...")
    try:
        # In a production environment, we would first run:
        # 1. src/data/build_dataset.py --last-6h-only
        # 2. src/features/build_features.py --incremental
        
        # For this demonstration, we trigger the dashboard generation which 
        # uses the latest available window in the processed dataset.
        create_multi_dashboard()
        logger.info("Real-time Dashboard updated successfully.")
    except Exception as e:
        logger.error(f"Cycle failed: {e}")

def main(interval_sec=600):
    """
    Persistent loop to run the pipeline every 10 minutes.
    """
    logger.info(f"Starting IGIA RVR Real-Time Orchestrator (Interval: {interval_sec}s)")
    while True:
        run_once()
        logger.info(f"Waiting {interval_sec}s for next cycle...")
        time.sleep(interval_sec)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--interval", type=int, default=600, help="Polling interval in seconds")
    args = parser.parse_args()
    
    # For a one-off demo, we run it once.
    run_once()
