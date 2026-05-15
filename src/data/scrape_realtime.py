"""
scrape_realtime.py — Live Data Scraper for IGIA RVR Forecasting

Pulls real-time meteorological and RVR data from the IITM WiFEX portal
(https://ews.tropmet.res.in/wifex/) and assembles a raw observation buffer
for consumption by the preprocessing pipeline.

Data Sources:
    1. AWS Weather Station (aws.php)     → Temperature, RH, Wind, Pressure
    2. RVR Observations (observations.php via graph_visibility.php?runway=XX)
                                          → Per-zone RVR values
    3. Visibility (igia_visibility.php)   → Ambient visibility (metres)

Output:
    data/realtime/latest_buffer.parquet — Rolling 6-hour buffer of raw observations
"""

import re
import json
import logging
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests

try:
    from playwright.sync_api import sync_playwright
    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    PLAYWRIGHT_AVAILABLE = False

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]
BUFFER_PATH = ROOT / "data" / "realtime" / "latest_buffer.parquet"
BUFFER_PATH.parent.mkdir(parents=True, exist_ok=True)

BASE_URL = "https://ews.tropmet.res.in/wifex"
IST = timezone(timedelta(hours=5, minutes=30))

# Timeout & retry settings
REQUEST_TIMEOUT = 30   # seconds
MAX_RETRIES = 3
RETRY_DELAY = 5        # seconds between retries

# Runway codes on the observations page → consolidated zone names
RUNWAY_ZONE_MAP = {
    "RWY09(TDZ)":  "09_TDZ",
    "RWY27(TDZ)":  "27_TDZ",
    "RWY10(TDZ)":  "10_TDZ",
    "RWY28(TDZ)":  "28_TDZ",
    "RWY28(MID)":  "MID_2810",
    "RWY11(TDZ)":  "11_TDZ",
    "RWY11(BEG)":  "11_BEG",
    "RWY29(TDZ)":  "29_TDZ",
    "RWY29(BEG)":  "29_BEG",
    "RWY29(MID)":  "MID_2911",
}

# All 10 consolidated zones the model expects
CONSOLIDATED_ZONES = [
    "09_TDZ", "27_TDZ", "10_TDZ", "28_TDZ", "MID_2810",
    "11_TDZ", "11_BEG", "29_TDZ", "29_BEG", "MID_2911",
]

# Mapping for the new live-rvr portal
NEW_PORTAL_ZONE_MAP = {
    "RWY09TDZ": "09_TDZ",
    "RWY27TDZ": "27_TDZ",
    "RWY10TDZ": "10_TDZ",
    "RWY28TDZ": "28_TDZ",
    "RWY28MID": "MID_2810",
    "RWY11R TDZ": "11_TDZ",
    "RWY11R BEG": "11_BEG",
    "RWY29L TDZ": "29_TDZ",
    "RWY29L BEG": "29_BEG",
    "RWY29L MID": "MID_2911",
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("scrape_realtime")

# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------
_session = requests.Session()
_session.headers.update({
    "User-Agent": "IGIA-RVR-Scraper/1.0 (Research; +https://github.com/alwyndev)",
    "Accept": "text/html,application/json",
    "Accept-Language": "en-IN,en;q=0.9",
})


def _fetch(url: str, *, as_json: bool = False, timeout: int = REQUEST_TIMEOUT):
    """Fetch a URL with retry logic.  Returns text or parsed JSON."""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = _session.get(url, timeout=timeout)
            resp.raise_for_status()
            return resp.json() if as_json else resp.text
        except requests.RequestException as exc:
            logger.warning("Attempt %d/%d failed for %s: %s", attempt, MAX_RETRIES, url, exc)
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY)
    logger.error("All %d attempts failed for %s", MAX_RETRIES, url)
    return None


# ---------------------------------------------------------------------------
# Chart.js parser  (embedded in aws.php / igia_visibility.php)
# ---------------------------------------------------------------------------
def _extract_chartjs_data(html: str, var_name: str) -> Optional[Dict]:
    """
    Extract a Chart.js data object from inline <script> in the HTML.

    These look like:
        const datatemp = {
            labels: ["2026-05-14 00:00:00", ...],
            datasets: [{
                label: 'Temperature 2m',
                data: [22.3, 22.1, ...],
                ...
            }, ...]
        };

    Returns a dict  {label_name: pd.Series(data, index=timestamps)} for each dataset.
    """
    # Strategy: find the var declaration start, then use bracket counting
    # to locate the matching closing brace (handles nested objects).
    start_pat = rf'(?:const|var|let)\s+{re.escape(var_name)}\s*=\s*\{{'
    m = re.search(start_pat, html)
    if not m:
        logger.debug("Chart.js var '%s' not found in HTML (%d chars)", var_name, len(html))
        return None

    # Find matching closing brace via bracket counting
    brace_start = m.end() - 1   # position of the opening '{'
    depth, pos = 0, brace_start
    for pos in range(brace_start, min(brace_start + 500_000, len(html))):
        ch = html[pos]
        if ch == '{':
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0:
                break
    block = html[m.start(): pos + 1]

    # --- Extract labels ---
    labels_match = re.search(r'labels\s*:\s*\[(.*?)\]', block, re.DOTALL)
    if not labels_match:
        logger.debug("No labels array found in '%s'", var_name)
        return None
    labels_raw = labels_match.group(1)
    timestamps = re.findall(r'"([^"]+)"', labels_raw)
    if not timestamps:
        timestamps = re.findall(r"'([^']+)'", labels_raw)
    if not timestamps:
        logger.debug("Empty labels in '%s'", var_name)
        return None
    index = pd.to_datetime(timestamps, errors="coerce")

    # --- Extract each dataset ---
    result = {}
    dataset_blocks = re.findall(
        r"\{\s*label\s*:\s*'([^']+)'[\s\S]*?data\s*:\s*\[([\s\S]*?)\]",
        block,
    )
    for label, data_str in dataset_blocks:
        values = []
        for v in re.findall(r'[\d.eE+-]+|null|NaN', data_str):
            if v.lower() in ("null", "nan"):
                values.append(np.nan)
            else:
                try:
                    values.append(float(v))
                except ValueError:
                    values.append(np.nan)
        # Align lengths
        n = min(len(index), len(values))
        result[label] = pd.Series(values[:n], index=index[:n], name=label)

    return result if result else None


# ---------------------------------------------------------------------------
# 1.  AWS Weather Station data
# ---------------------------------------------------------------------------
def _pick_best_series(data: Dict[str, pd.Series], preferred_heights: list) -> Optional[pd.Series]:
    """
    Pick the best dataset from a Chart.js extraction result.

    Prefers heights in the order given (e.g., ["10m", "20m", "2m"]).
    Skips any series that is entirely NaN, falling back to the next height.
    """
    if not data:
        return None

    for height in preferred_heights:
        for label, series in data.items():
            if height.lower() in label.lower():
                if series.notna().any():
                    return series
                else:
                    logger.debug("  %s is all-NaN, trying next height", label)

    # Ultimate fallback: first series with any non-NaN data
    for series in data.values():
        if series.notna().any():
            return series

    return None


def scrape_aws() -> pd.DataFrame:
    """
    Scrape weather station data from aws.php.

    Returns a DataFrame indexed by datetime with columns:
        temp_c, rh_pct, wind_speed_ms, wind_dir, pressure_hpa

    Height preference order: 10m > 20m > 2m (skips all-NaN sensors).
    """
    logger.info("Scraping AWS weather station data...")
    html = _fetch(f"{BASE_URL}/aws.php")
    if html is None:
        logger.error("Failed to fetch aws.php")
        return pd.DataFrame()

    records = {}

    # --- Temperature (prefer 2m) ---
    temp_data = _extract_chartjs_data(html, "datatemp")
    s = _pick_best_series(temp_data, ["2m", "5m", "10m", "20m"])
    if s is not None:
        records["temp_c"] = s

    # --- Relative Humidity (prefer 2m) ---
    rh_data = _extract_chartjs_data(html, "dataRh")
    s = _pick_best_series(rh_data, ["2m", "5m", "10m", "20m"])
    if s is not None:
        records["rh_pct"] = s

    # --- Wind Speed (prefer 10m, fall back to 20m then 2m) ---
    ws_data = _extract_chartjs_data(html, "dataWs")
    s = _pick_best_series(ws_data, ["10m", "20m", "2m"])
    if s is not None:
        records["wind_speed_ms"] = s

    # --- Wind Direction (prefer 10m, fall back to 20m then 2m) ---
    wd_data = _extract_chartjs_data(html, "dataWd")
    s = _pick_best_series(wd_data, ["10m", "20m", "2m"])
    if s is not None:
        records["wind_dir"] = s

    # --- Pressure (no dedicated chart on current AWS page) ---
    pr_data = _extract_chartjs_data(html, "dataPressure")
    if pr_data:
        # Just take the first dataset
        records["pressure_hpa"] = next(iter(pr_data.values()))

    if not records:
        logger.warning("No AWS data extracted!")
        return pd.DataFrame()

    df = pd.DataFrame(records)
    df.index.name = "datetime"

    logger.info("AWS data: %d rows, columns: %s", len(df), list(df.columns))
    return df


# ---------------------------------------------------------------------------
# 2.  Visibility data
# ---------------------------------------------------------------------------
def scrape_visibility() -> pd.DataFrame:
    """
    Scrape ambient visibility from igia_visibility.php.

    Returns a DataFrame with column 'visibility_m' indexed by datetime.
    """
    logger.info("Scraping visibility data...")
    html = _fetch(f"{BASE_URL}/igia_visibility.php")
    if html is None:
        logger.error("Failed to fetch igia_visibility.php")
        return pd.DataFrame()

    vis_data = _extract_chartjs_data(html, "dataVis")
    if vis_data:
        # Try to find the "Visibility" dataset
        for label, series in vis_data.items():
            if "visibility" in label.lower() or "vis" in label.lower():
                return pd.DataFrame({"visibility_m": series})
        # Fallback: first dataset
        return pd.DataFrame({"visibility_m": next(iter(vis_data.values()))})

    # Alternative: try other variable names
    for var in ["data_vis", "dataVisibility", "data"]:
        vis_data = _extract_chartjs_data(html, var)
        if vis_data:
            return pd.DataFrame({"visibility_m": next(iter(vis_data.values()))})

    logger.warning("No visibility data extracted!")
    return pd.DataFrame()


# ---------------------------------------------------------------------------
# 3.  Per-zone RVR data  (via Highcharts JSON endpoint)
# ---------------------------------------------------------------------------
def scrape_rvr_zone(runway_code: str) -> pd.DataFrame:
    """
    Fetch RVR for a single runway zone from graph_visibility.php.

    The endpoint returns JSON:  [[timestamp_ms, rvr_value], ...]
    """
    url = f"{BASE_URL}/graph_visibility.php?runway={runway_code}"
    data = _fetch(url, as_json=True)
    if data is None:
        return pd.DataFrame()

    if not isinstance(data, list) or len(data) == 0:
        logger.debug("Empty RVR data for %s", runway_code)
        return pd.DataFrame()

    # Parse [[epoch_ms, value], ...]
    timestamps, values = [], []
    for entry in data:
        if isinstance(entry, (list, tuple)) and len(entry) >= 2:
            ts_ms = entry[0]
            val = entry[1]
            try:
                dt = datetime.fromtimestamp(ts_ms / 1000, tz=IST)
                timestamps.append(dt)
                values.append(float(val) if val is not None else np.nan)
            except (ValueError, TypeError, OSError):
                continue

    if not timestamps:
        return pd.DataFrame()

    zone_name = RUNWAY_ZONE_MAP.get(runway_code, runway_code)
    df = pd.DataFrame(
        {f"{zone_name}_rvr_actual_mean": values},
        index=pd.DatetimeIndex(timestamps, name="datetime"),
    )
    return df


def _parse_rvr_value(val_str: str) -> float:
    """Parse 'P2000 m' or '1500 m' into a float. Returns NaN for '--- m' or 'NA'."""
    val_str = val_str.upper().replace(" M", "").strip()
    if val_str in ("---", "NA", ""):
        return np.nan
    # Handle 'P2000' (Plus 2000) -> 2000
    if val_str.startswith("P") or val_str.startswith("M"):
        val_str = val_str[1:]
    try:
        return float(val_str)
    except ValueError:
        return np.nan

def scrape_rvr_new_portal() -> pd.DataFrame:
    """
    Scrape RVR using Playwright from the new live-rvr portal.
    Requires playwright to be installed.
    """
    if not PLAYWRIGHT_AVAILABLE:
        logger.warning("Playwright not available. Falling back to WiFEX RVR scraping.")
        return pd.DataFrame()

    logger.info("Scraping RVR from new portal (http://rvrcamd.imd.gov.in:5000/live-rvr)...")
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            
            page.goto("http://rvrcamd.imd.gov.in:5000/live-rvr", timeout=30000)
            page.wait_for_timeout(2000)
            
            # Select New Delhi Airport
            page.click("text=Select Airport")
            page.wait_for_timeout(1000)
            page.click("text=New Delhi Airport")
            
            # Wait for websocket data to populate (about 5s)
            page.wait_for_timeout(5000)
            
            text = page.locator("body").inner_text()
            browser.close()
            
            # Parse the text block
            lines = [line.strip() for line in text.split("\n") if line.strip()]
            
            rvr_data = {}
            for i, line in enumerate(lines):
                # Check if this line is a runway label we care about
                if line in NEW_PORTAL_ZONE_MAP and i + 1 < len(lines):
                    zone = NEW_PORTAL_ZONE_MAP[line]
                    val_str = lines[i + 1]
                    val = _parse_rvr_value(val_str)
                    rvr_data[f"{zone}_rvr_actual_mean"] = val
                    logger.debug("  Found %s -> %s (Parsed: %s)", line, val_str, val)
                    
            if not rvr_data:
                logger.warning("No recognizable RVR data found in the new portal text.")
                return pd.DataFrame()
                
            # Create a 1-row DataFrame with the current time
            now = datetime.now(IST).replace(second=0, microsecond=0)
            df = pd.DataFrame([rvr_data], index=pd.DatetimeIndex([now], name="datetime"))
            
            logger.info("Successfully scraped %d zones from new portal.", len(rvr_data))
            return df
            
    except Exception as e:
        logger.error("Error scraping new RVR portal: %s", e)
        return pd.DataFrame()

def scrape_all_rvr() -> pd.DataFrame:
    """
    Scrape RVR for all 10 consolidated zones.
    First tries the new live-rvr portal. If it fails/empty, falls back to WiFEX.
    """
    logger.info("Scraping RVR data for all zones...")
    
    # Attempt new portal first
    df_new = scrape_rvr_new_portal()
    if not df_new.empty:
        # Ensure all 10 zone columns exist
        for zone in CONSOLIDATED_ZONES:
            col = f"{zone}_rvr_actual_mean"
            if col not in df_new.columns:
                df_new[col] = np.nan
        return df_new
        
    logger.info("Falling back to WiFEX RVR scraping...")
    frames = []

    for rwy_code, zone_name in RUNWAY_ZONE_MAP.items():
        logger.info("  -> %s (%s)", rwy_code, zone_name)
        df_zone = scrape_rvr_zone(rwy_code)
        if not df_zone.empty:
            frames.append(df_zone)
        time.sleep(0.5)  # Be polite to the server

    if not frames:
        logger.warning("No RVR data retrieved for any zone!")
        return pd.DataFrame()

    # Merge all zones on their datetime index
    df = frames[0]
    for other in frames[1:]:
        df = df.join(other, how="outer")

    # Ensure all 10 zone columns exist
    for zone in CONSOLIDATED_ZONES:
        col = f"{zone}_rvr_actual_mean"
        if col not in df.columns:
            logger.warning("Zone %s missing from RVR data -- padding with NaN", zone)
            df[col] = np.nan

    logger.info("RVR data: %d rows, %d zone columns", len(df), len([c for c in df.columns if "rvr_actual_mean" in c]))
    return df


# ---------------------------------------------------------------------------
# 4.  Assemble & save the raw buffer
# ---------------------------------------------------------------------------
def assemble_buffer() -> pd.DataFrame:
    """
    Pull data from all sources, merge on datetime index, and save to Parquet.
    """
    logger.info("=" * 60)
    logger.info("IGIA Real-Time Data Scraper -- %s IST", datetime.now(IST).strftime("%Y-%m-%d %H:%M"))
    logger.info("=" * 60)

    # Scrape all sources
    df_aws = scrape_aws()
    df_vis = scrape_visibility()
    df_rvr = scrape_all_rvr()

    # Start with the source that has the most data
    all_frames = [f for f in [df_rvr, df_aws, df_vis] if not f.empty]
    if not all_frames:
        logger.error("No data scraped from any source! The portal may be offline.")
        return pd.DataFrame()

    # Ensure all indices are timezone-naive for merging
    for i, f in enumerate(all_frames):
        if f.index.tz is not None:
            all_frames[i] = f.tz_localize(None) if f.index.tz is None else f.tz_convert(IST).tz_localize(None)

    # Merge on datetime
    df = all_frames[0]
    for other in all_frames[1:]:
        df = df.join(other, how="outer")

    # Sort by time
    df.sort_index(inplace=True)

    # Remove duplicate timestamps (keep last)
    df = df[~df.index.duplicated(keep="last")]

    # If an existing buffer exists, append new data (rolling 12-hour window)
    if BUFFER_PATH.exists():
        try:
            df_old = pd.read_parquet(BUFFER_PATH)
            df = pd.concat([df_old, df])
            df = df[~df.index.duplicated(keep="last")]
            df.sort_index(inplace=True)

            # Keep only the last 12 hours of data
            cutoff = df.index.max() - pd.Timedelta(hours=12)
            df = df.loc[df.index >= cutoff]
        except Exception as e:
            logger.warning("Could not merge with existing buffer: %s", e)

    # Save
    df.to_parquet(BUFFER_PATH)
    logger.info("Buffer saved to %s  (%d rows, %d cols)", BUFFER_PATH, len(df), len(df.columns))
    logger.info("Time range: %s -> %s", df.index.min(), df.index.max())

    return df


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
def main():
    """Run the scraper and print a summary."""
    df = assemble_buffer()

    if df.empty:
        print("\n[!] No data scraped. The WiFEX portal may be offline or in off-season mode.")
        print("    The portal displays: 'Fog forecast has been stopped and will be")
        print("    continued during next winter season.'")
        print("\n    During off-season, use historical data from:")
        print(f"    {ROOT / 'data' / 'processed' / 'igia_rvr_training_dataset_multi.parquet'}")
        return

    print(f"\n{'='*60}")
    print(f"  Scrape Summary")
    print(f"{'='*60}")
    print(f"  Time range : {df.index.min()} -> {df.index.max()}")
    print(f"  Rows       : {len(df)}")
    print(f"  Columns    : {len(df.columns)}")
    print(f"  Buffer     : {BUFFER_PATH}")
    print()

    # Per-column completeness
    print("  Column Completeness:")
    for col in sorted(df.columns):
        pct = (1 - df[col].isna().mean()) * 100
        filled = int(pct // 5)
        bar = "#" * filled + "." * (20 - filled)
        print(f"    {col:<35s} [{bar}] {pct:5.1f}%")

    print(f"\n  Next step: Run preprocess_realtime.py to prepare model input.")


if __name__ == "__main__":
    main()
