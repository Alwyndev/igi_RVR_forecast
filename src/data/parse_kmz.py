"""
parse_kmz.py — Extracts sensor coordinates from the MET_ANTENNA.kmz file.
Saves them to a JSON mapping for use in spatial interpolation later.
"""

import os
import json
import zipfile
from fastkml import kml
import logging
from .runway_config import CONSOLIDATED_ZONES

logger = logging.getLogger(__name__)

def extract_kmz_coordinates(kmz_path: str, output_json: str):
    """Extracts coordinates using robust regex and fuzzy mapping."""
    logger.info(f"Extracting coordinates from {kmz_path}...")
    
    with zipfile.ZipFile(kmz_path, 'r') as kmz:
        kml_name = [name for name in kmz.namelist() if name.endswith('.kml')][0]
        kml_data = kmz.read(kml_name).decode('utf-8', 'ignore')
        
    import re
    # Match <Placemark> blocks
    placemarks = re.findall(r'<Placemark.*?>.*?</Placemark>', kml_data, re.DOTALL)
    
    raw_coords = {}
    for pm in placemarks:
        name_match = re.search(r'<name>(.*?)</name>', pm)
        coord_match = re.search(r'<coordinates>(.*?)</coordinates>', pm)
        if name_match and coord_match:
            name = name_match.group(1).strip()
            # Coordinates format: lon,lat,alt
            parts = coord_match.group(1).strip().split(',')
            if len(parts) >= 2:
                raw_coords[name] = (float(parts[1]), float(parts[0])) # lat, lon

    # Fuzzy Mapping to Consolidated Zones
    final_coords = {}
    zone_points = {z: [] for z in CONSOLIDATED_ZONES}
    
    for name, (lat, lon) in raw_coords.items():
        name_up = name.upper()
        target_zone = None
        
        # Strip A (09/27)
        if "09_TD" in name_up or "09_RWY" in name_up: target_zone = "09_TDZ"
        elif "27_TD" in name_up or "27_RWY" in name_up: target_zone = "27_TDZ"
        # Strip B (10/28)
        elif "10_TD" in name_up or "10_B" in name_up or "10_OLD" in name_up: target_zone = "10_TDZ"
        elif "28_TD" in name_up or "28_B" in name_up or "28_OLD" in name_up: target_zone = "28_TDZ"
        elif "28_MID" in name_up or "10_MID" in name_up or "MID2810" in name_up: target_zone = "MID_2810"
        # Strip C/D (11/29)
        elif "11L_TD" in name_up or "11_TD" in name_up: target_zone = "11_TDZ"
        elif "11L_B" in name_up or "11_B" in name_up: target_zone = "11_BEG"
        elif "29R_TD" in name_up or "29_TD" in name_up: target_zone = "29_TDZ"
        elif "29R_B" in name_up or "29_B" in name_up: target_zone = "29_BEG"
        elif "29L_MID" in name_up or "29R_MID" in name_up or "MID2911" in name_up: target_zone = "MID_2911"
        
        if target_zone:
            zone_points[target_zone].append((lat, lon))
            
    # Average the points for each consolidated zone
    for zone, points in zone_points.items():
        if points:
            lats, lons = zip(*points)
            final_coords[zone] = {"lat": sum(lats)/len(lats), "lon": sum(lons)/len(lons)}
            
    logger.info(f"Finalized coordinates for {len(final_coords)} zones.")
    with open(output_json, 'w') as f:
        json.dump(final_coords, f, indent=4)

        
if __name__ == "__main__":
    import sys
    root = sys.argv[1] if len(sys.argv) > 1 else r"c:\Users\alwyn\OneDrive\Desktop\IGI_Antigravity"
    kmz = os.path.join(root, "Latest Data", "MET_ANTENNA.kmz")
    out = os.path.join(root, "data", "interim", "sensor_coordinates.json")
    extract_kmz_coordinates(kmz, out)
