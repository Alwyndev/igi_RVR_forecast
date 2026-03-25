"""
runway_config.py — Physical Runway Strip Mapping for IGIA

IGIA has 4 physical strips with 8 logical runway directions. Sensors at each
strip are at TDZ (touchdown zone), BEG (beginning), and MID (midpoint).
The MID sensor is shared between both directions of a strip.

This module encodes the spatial mapping so the BiLSTM model can correctly
share MID-point features across runway pairs.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

# ============================================================================
# Zone IDs: Canonical 0-indexed IDs for the 13 sensor positions
# These are the output indices of the BiLSTM's 13-element prediction vector.
# ============================================================================
ZONE_IDS = {
    "09_TDZ": 0,
    "27_TDZ": 1,
    "10_TDZ": 2,
    "28_TDZ": 3,
    "MID_2810": 4,
    "11_TDZ": 5,
    "11_BEG": 6,
    "29_TDZ": 7,
    "29_BEG": 8,
    "MID_2911": 9,
    "10_NEW_TDZ": 10,  # Same physical location as 10_TDZ (sensor upgrade)
    "28_NEW_TDZ": 11,  # Same physical location as 28_TDZ (sensor upgrade)
    "MID_2810_NEW": 12, # Same physical location as MID_2810 (sensor upgrade)
}

# Reverse lookup
ZONE_NAMES = {v: k for k, v in ZONE_IDS.items()}


# ============================================================================
# Physical Strip Definitions
# ============================================================================
@dataclass
class RunwayStrip:
    """Represents one physical runway strip at IGIA."""
    name: str
    strip_id: str  # A, B, C, D
    directions: Tuple[str, str]  # e.g., ("09", "27")
    sensors: Dict[str, str]  # position_type -> zone_name
    old_new_mapping: Optional[Dict[str, str]] = field(default_factory=dict)

STRIPS = {
    "A": RunwayStrip(
        name="Strip A (09/27)",
        strip_id="A",
        directions=("09", "27"),
        sensors={
            "TDZ_FWD": "09_TDZ",     # 09 end
            "TDZ_REV": "27_TDZ",     # 27 end
            # Note: No dedicated MID folder found for Strip A in the data
        },
    ),
    "B": RunwayStrip(
        name="Strip B (10/28)",
        strip_id="B",
        directions=("10", "28"),
        sensors={
            "TDZ_FWD": "10_TDZ",
            "TDZ_REV": "28_TDZ",
            "MID": "MID_2810",
        },
        old_new_mapping={
            "10_TDZ": "10_NEW_TDZ",       # OLD -> NEW
            "28_TDZ": "28_NEW_TDZ",       # OLD -> NEW
            "MID_2810": "MID_2810_NEW",   # OLD -> NEW
        },
    ),
    "C": RunwayStrip(
        name="Strip C (11/29R)",
        strip_id="C",
        directions=("11", "29R"),
        sensors={
            "TDZ_FWD": "11_TDZ",
            "BEG_FWD": "11_BEG",
            # MID -> MID_2911 is shared with Strip D conceptually
        },
    ),
    "D": RunwayStrip(
        name="Strip D (29L/11R)",
        strip_id="D",
        directions=("29L", "11R"),
        sensors={
            "TDZ_FWD": "29_TDZ",
            "BEG_FWD": "29_BEG",
            "MID": "MID_2911",
        },
    ),
}


# ============================================================================
# Data Folder → Zone Mapping
# Maps each data folder name to its canonical zone ID
# ============================================================================

# For Latest Data/RVR/ (2019–2023)
RVR_FOLDER_TO_ZONE = {
    "RWY09":         "09_TDZ",
    "RWY27":         "27_TDZ",
    "RWY10":         "10_TDZ",
    "RWY28":         "28_TDZ",
    "RWY10_N":       "10_NEW_TDZ",
    "RWY28_N":       "28_NEW_TDZ",
    "RWY11":         "11_TDZ",
    "RWY11beg":      "11_BEG",
    "RWY29":         "29_TDZ",
    "RWY29beg":      "29_BEG",
    "RWYMID2810":    "MID_2810",
    "RWYMID2810_N":  "MID_2810_NEW",
    "RWYMID2911":    "MID_2911",
}

# For Latest Data/RVR DATA 2024-25/ (2024–2025)
RVR_2024_FOLDER_TO_ZONE = {
    "09 TDZ":       "09_TDZ",
    "27 TDZ":       "27_TDZ",
    "10 OLD":       "10_TDZ",
    "28 OLD":       "28_TDZ",
    "10 NEW":       "10_NEW_TDZ",
    "28 NEW":       "28_NEW_TDZ",
    "11 TDZ":       "11_TDZ",
    "11 BEG":       "11_BEG",
    "29 TDZ":       "29_TDZ",
    "29 BEG":       "29_BEG",
    "28 MID OLD":   "MID_2810",
    "28 MID NEW":   "MID_2810_NEW",
    "29L MID":      "MID_2911",
}


# ============================================================================
# OLD/NEW Sensor Consolidation
# For model training, OLD and NEW refer to the same physical location.
# We consolidate them into a single zone for the unified feature matrix.
# ============================================================================

# Maps NEW zone names to their OLD counterpart (same physical location)
NEW_TO_OLD_ZONE = {
    "10_NEW_TDZ":   "10_TDZ",
    "28_NEW_TDZ":   "28_TDZ",
    "MID_2810_NEW": "MID_2810",
}

# After consolidation, the model operates on 10 unique spatial positions:
CONSOLIDATED_ZONES = [
    "09_TDZ",      # Strip A - end 1
    "27_TDZ",      # Strip A - end 2
    "10_TDZ",      # Strip B - end 1 (OLD+NEW consolidated)
    "28_TDZ",      # Strip B - end 2 (OLD+NEW consolidated)
    "MID_2810",    # Strip B - midpoint (OLD+NEW consolidated)
    "11_TDZ",      # Strip C - TDZ
    "11_BEG",      # Strip C - BEG
    "29_TDZ",      # Strip D - TDZ
    "29_BEG",      # Strip D - BEG
    "MID_2911",    # Strip C/D - shared midpoint
]

# Consolidated zone count (this is the model output dimension)
NUM_ZONES = len(CONSOLIDATED_ZONES)  # 10

# Zone index lookup for the consolidated zones
CONSOLIDATED_ZONE_IDS = {name: i for i, name in enumerate(CONSOLIDATED_ZONES)}


# ============================================================================
# RVR File Column Schema
# ============================================================================
RVR_COLUMNS = [
    "time",
    "rvr_limited",
    "mor_limited",
    "rvr_actual",
    "mor_actual",
    "blm",           # Background Luminance Meter reading
    "transmissivity", # Trf - transmissivity
    "ref_voltage",    # Ref(v) - reference voltage
    "pd_voltage",     # PD(v) - photo-detector voltage
]

# Data types for RVR columns (time is parsed separately)
RVR_NUMERIC_COLS = [
    "rvr_limited", "mor_limited", "rvr_actual", "mor_actual",
    "blm", "transmissivity", "ref_voltage", "pd_voltage"
]


def get_data_source_for_year(year: int) -> str:
    """Determine the canonical data source folder for a given year.

    Returns:
        'RVR' for 2019-2023, 'RVR_2024_25' for 2024-2025
    """
    if year <= 2023:
        return "RVR"
    else:
        return "RVR_2024_25"


def consolidate_zone_name(zone: str) -> str:
    """Map a NEW sensor zone name to its consolidated (OLD) counterpart.

    If the zone is already a consolidated name, returns it unchanged.
    """
    return NEW_TO_OLD_ZONE.get(zone, zone)


if __name__ == "__main__":
    print(f"IGIA Runway Configuration")
    print(f"  Physical strips: {len(STRIPS)}")
    print(f"  Raw sensor zones: {len(ZONE_IDS)}")
    print(f"  Consolidated zones (model output dim): {NUM_ZONES}")
    print()
    for sid, strip in STRIPS.items():
        print(f"  {strip.name}: sensors={list(strip.sensors.values())}")
        if strip.old_new_mapping:
            print(f"    OLD→NEW: {strip.old_new_mapping}")
