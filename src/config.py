from pathlib import Path

# =============================================================================
# 1. PROJECT ROOT CONFIGURATION
# =============================================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent


# =============================================================================
# 2. DATA DIRECTORY STRUCTURE
# =============================================================================
DATA_DIR = PROJECT_ROOT / "data"

# --- INPUTS (Read-Only Zone) ---
# Root directory for inputs
INPUT_DIR = DATA_DIR / "input"

# Raw files downloaded from the API
RAW_DIR = INPUT_DIR / "raw"

# Simplified files (Raw material for the pipeline)
# Source HDF5 files are stored here
SIMPLIFIED_DIR = INPUT_DIR / "simplified_raw"


# --- OUTPUTS (Read-Write Zone) ---
# Root directory for all production outputs
OUTPUT_DIR = DATA_DIR / "output"

# Data Lake: Processed, interpolated, and enriched HDF5 files (Intermediate)
# (Formerly 'processed' folder)
PROCESSED_DIR = OUTPUT_DIR / "processed_lake"

# Final Machine Learning datasets (.npz)
DATASETS_DIR = OUTPUT_DIR / "datasets"


# =============================================================================
# 3. LOGGING AND OUTPUTS
# =============================================================================
LOGS_DIR = PROJECT_ROOT / "logs"
TENSORBOARD_LOGS_DIR = LOGS_DIR / "tensorboard"
GRID_SEARCH_RESULTS_DIR = LOGS_DIR / "grid_search"


# =============================================================================
# 4. BUSINESS LOGIC & CONFIGURATION
# =============================================================================
API_BASE_URL = "https://crl.cassiopee.studio/"


# Sentinel value for padding (physically impossible value)
# Used to distinguish missing data from real zero values (e.g., Touchdown height)
PADDING_VALUE = -1.0e9


# CRITICAL CONFIGURATION:
# This file dictates strictly which parameters are extracted from RAW to SIMPLIFIED.
# It is now located in the input folder as it is a user-provided constraint.
SELECTED_PARAMS_FILE = INPUT_DIR / "selected_parameters.txt"


# =============================================================================
# 5. UTILITIES
# =============================================================================
def ensure_directories() -> None:
    """
    Creates the complete directory tree if it does not exist.
    """
    directories = [
        DATA_DIR,
        INPUT_DIR,
        OUTPUT_DIR,
        RAW_DIR,
        SIMPLIFIED_DIR,
        PROCESSED_DIR,
        DATASETS_DIR,
        LOGS_DIR,
        TENSORBOARD_LOGS_DIR,
        GRID_SEARCH_RESULTS_DIR
    ]

    for directory in directories:
        if not directory.exists():
            print(f"[INFO] Creating directory: {directory}")
            directory.mkdir(parents=True, exist_ok=True)

if __name__ == "__main__":
    print(f"[INFO] Project Root: {PROJECT_ROOT}")
    ensure_directories()
    print("[INFO] Directory structure updated.")