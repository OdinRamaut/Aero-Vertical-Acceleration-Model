import sys
import logging
import time
import os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Tuple

# =============================================================================
# 1. PROJECT SETUP
# =============================================================================
# Add project root to sys.path to ensure imports work correctly
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.config import SIMPLIFIED_DIR, PROCESSED_DIR, LOGS_DIR
from src.processing.ExtractTransformLoad.resampling import FlightDataTransformer
from src.processing.features.engineering import FeatureEngineer

# =============================================================================
# 2. GLOBAL CONFIGURATION
# =============================================================================
# Target sampling frequency for the temporal grid (Hz)
TARGET_FREQ = 4.0

# Logger initialization (Basic setup, handlers added in main)
logger = logging.getLogger(__name__)


def setup_logging():
    """
    Configures logging to file and console with robust handling.
    """
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    log_file = LOGS_DIR / "processing_pipeline.log"

    # Reset handlers to avoid duplication or missing initial logs
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

    # File Handler
    file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Console Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return log_file


def process_single_flight(input_path: Path) -> Tuple[str, str]:
    """
    Worker function to process a single flight file in a separate process.

    Pipeline Steps:
    1. Transformation: Resampling and alignment to TARGET_FREQ (FlightDataTransformer).
    2. Engineering: Physics-based feature extraction (FeatureEngineer).

    Args:
        input_path (Path): Path to the simplified HDF5 file.

    Returns:
        Tuple[str, str]: Status ("SUCCESS", "error_message") and the filename.
    """
    try:
        file_name = input_path.name
        # Define output path in the Data Lake (Processed Zone)
        # Naming convention: processed_{original_name}
        output_path = PROCESSED_DIR / f"processed_{file_name}"

        # --- Step 0: Enforce Overwrite ---
        if output_path.exists():
            output_path.unlink()

        # --- Step 1: Transformation (Resampling) ---
        # Aligns raw data to a common temporal grid
        transformer = FlightDataTransformer(target_freq=TARGET_FREQ)

        # The transformer creates the new file at output_path
        result_path = transformer.process_flight(input_path=input_path, output_path=output_path)

        if result_path is None:
            return "SKIPPED", f"Transformation returned None for {file_name}"

        # --- Step 2: Feature Engineering (Physics) ---
        # Operates in-place on the newly created processed file
        engineer = FeatureEngineer(sampling_freq=TARGET_FREQ)
        engineer.process_file(output_path)

        return "SUCCESS", file_name

    except Exception as e:
        # Catch-all to prevent one failure from stopping the whole batch
        return "ERROR", f"{input_path.name}: {str(e)}"


def main():
    # Setup logging immediately upon starting main
    log_file = setup_logging()

    logger.info("========================================")
    logger.info("   STARTING DATA PROCESSING PIPELINE    ")
    logger.info("========================================")
    logger.info(f"Logs writing to: {log_file}")
    logger.info(f"Configuration: Freq={TARGET_FREQ}Hz")
    logger.info(f"Input Directory:  {SIMPLIFIED_DIR}")
    logger.info(f"Output Directory: {PROCESSED_DIR}")

    # Ensure output directory exists
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Discovery Phase
    # We look for all .hdf5 files in the simplified directory
    files = list(SIMPLIFIED_DIR.glob("*.hdf5"))
    total_files = len(files)

    if total_files == 0:
        logger.warning(f"No files found in {SIMPLIFIED_DIR}. Please run '01_download_data.py' first.")
        return

    logger.info(f"Found {total_files} flights to process.")

    # 2. Parallel Execution Strategy
    # We use ProcessPoolExecutor to bypass the Global Interpreter Lock (GIL)
    # This is efficient for CPU-bound tasks like interpolation.
    # We leave 1 core free for system stability.
    max_workers = max(1, os.cpu_count() - 1)
    logger.info(f"Allocating {max_workers} worker processes...")

    start_time = time.time()
    stats = {"SUCCESS": 0, "SKIPPED": 0, "ERROR": 0}

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_file = {executor.submit(process_single_flight, f): f for f in files}

        # Process results as they complete (as_completed)
        for i, future in enumerate(as_completed(future_to_file), 1):
            status, message = future.result()

            if status == "SUCCESS":
                stats["SUCCESS"] += 1
            elif status == "SKIPPED":
                stats["SKIPPED"] += 1
                logger.warning(f"[SKIP] {message}")
            else:
                stats["ERROR"] += 1
                logger.error(f"[FAIL] {message}")

            # Progress logging every 10 files
            if i % 10 == 0 or i == total_files:
                elapsed = time.time() - start_time
                rate = i / elapsed if elapsed > 0 else 0
                logger.info(f"Progress: {i}/{total_files} ({rate:.2f} files/s) - "
                            f"Success: {stats['SUCCESS']} | Errors: {stats['ERROR']}")

    # 3. Final Summary
    duration = time.time() - start_time
    logger.info("========================================")
    logger.info("          PROCESSING COMPLETE           ")
    logger.info("========================================")
    logger.info(f"Total Time: {duration:.2f}s")
    logger.info(f"Processed: {stats['SUCCESS']}")
    logger.info(f"Failed:    {stats['ERROR']}")
    logger.info(f"Skipped:   {stats['SKIPPED']}")
    logger.info(f"Data Lake Location: {PROCESSED_DIR}")


if __name__ == "__main__":
    main()