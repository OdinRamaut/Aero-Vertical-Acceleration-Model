import sys
import logging
import shutil
from pathlib import Path
import h5py
from collections import defaultdict

# =============================================================================
# 1. PROJECT SETUP
# =============================================================================
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.config import PROCESSED_DIR, DATASETS_DIR, LOGS_DIR
from src.processing.building.builder import DatasetBuilder

from src.processing.building.slicing import (
    ModularFlightSlicer,
    ThresholdCondition,
    EventOffsetCondition
)

# =============================================================================
# 2. EXPERIMENT CONFIGURATION (USER ZONE)
# =============================================================================
EXPERIMENT_CONFIG = {
    "experiment_name": "Exp_07_Landing_VRTG_MORE_FEATURES",
    "target": "VRTG_C",
    "extraction_method": "max",
    "test_size": 0.2,
    "target_phase": [11, 12, 13],
    "val_size": 0.2,
    "random_state": 42,
    "features": {
        "strategy": "ALL_FEATURES",
        "exclude": ['PITCH_CMD_CPT_C', 'PITCH_CMD_FO_C', 'PITCH_RATE_C', 'ROLL_CMD_CPT_C', 'ROLL_CMD_FO_C', 'RUDD_PED_POS_C', 'IVVR', ]  # User-defined exclusions
    }
}

# =============================================================================
# 3. LOGGING CONFIGURATION (Dual Output)
# =============================================================================
# Create a custom logger
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)  # Capture everything globally

# A. Console Handler (Clean: Only INFO and above)
c_handler = logging.StreamHandler(sys.stdout)
c_handler.setLevel(logging.INFO)
c_format = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s', datefmt='%H:%M:%S')
c_handler.setFormatter(c_format)
logger.addHandler(c_handler)

# B. File Handler (Verbose: DEBUG and above)
# Create experiment-specific log directory
exp_log_dir = LOGS_DIR / EXPERIMENT_CONFIG["experiment_name"]
exp_log_dir.mkdir(parents=True, exist_ok=True)

log_file = exp_log_dir / "dataset_generation.log"
f_handler = logging.FileHandler(log_file, mode='w')  # 'w' overwrites each run
f_handler.setLevel(logging.DEBUG)
f_format = logging.Formatter('%(asctime)s [%(levelname)s] %(name)s - %(message)s')
f_handler.setFormatter(f_format)
logger.addHandler(f_handler)


# =============================================================================
# 4. MAIN PIPELINE
# =============================================================================
def main():
    logger.info("========================================")
    logger.info("      DATASET GENERATION PIPELINE       ")
    logger.info("========================================")
    logger.info(f"Logs will be saved to: {log_file}")

    # 1. Load Configuration
    config = EXPERIMENT_CONFIG
    exp_name = config["experiment_name"]

    logger.info(f"Experiment: {exp_name}")
    logger.info(f"Target:     {config['target']} (Method: {config.get('extraction_method', 'max')})")
    logger.info(f"Target Phase: {config.get('target_phase', 'Coupled with X')}")

    # --- 1b. Feature Auto-Discovery (Early Exclusion Strategy) ---
    raw_feats = config["features"]
    is_dynamic_mode = False
    user_exclusions = set()

    # Detect mode (String vs Dict)
    if raw_feats == "ALL_FEATURES":
        is_dynamic_mode = True
    elif isinstance(raw_feats, dict) and raw_feats.get("strategy") == "ALL_FEATURES":
        is_dynamic_mode = True
        user_exclusions = set(raw_feats.get("exclude", []))

    if is_dynamic_mode:
        logger.info(f"Detected 'ALL_FEATURES' strategy.")
        logger.info(f"User Exclusions ({len(user_exclusions)}): {sorted(list(user_exclusions))}")
        logger.info("Initiating detailed schema analysis (Exclusions applied A PRIORI)...")

        files = list(PROCESSED_DIR.glob("*.h5")) + list(PROCESSED_DIR.glob("*.hdf5"))
        if not files:
            logger.critical(f"FATAL: No data found in {PROCESSED_DIR}.")
            sys.exit(1)

        # 1. Mapping: Signature -> List of Filenames
        schema_map = defaultdict(list)

        # MERGING EXCLUSIONS: System + User
        # We ignore these variables DURING THE SCAN. This allows grouping files
        # that differ only by the presence/absence of 'IVVR' (or other exclusions).
        system_exclusions = {"Time", "MetaData", "FLIGHT_PHASE", config["target"]}
        all_exclusions = system_exclusions.union(user_exclusions)

        logger.info(f"Scanning {len(files)} files for feature signatures...")

        for fpath in files:
            try:
                with h5py.File(fpath, 'r') as f:
                    keys = set(f['Parameters'].keys())
                    # Creation of the effective signature (Immediate filtering)
                    effective_features = tuple(sorted([k for k in keys if k not in all_exclusions]))

                    if effective_features:
                        schema_map[effective_features].append(fpath.name)
            except Exception:
                logger.warning(f"[SKIP] Read error on {fpath.name}")

        if not schema_map:
            logger.critical("FATAL: Could not extract any valid feature signature.")
            sys.exit(1)

        # 2. Identify the Mode (Dominant Schema)
        dominant_signature = max(schema_map, key=lambda k: len(schema_map[k]))
        dominant_count = len(schema_map[dominant_signature])
        dominant_set = set(dominant_signature)

        total_valid_scans = sum(len(v) for v in schema_map.values())
        coverage_percent = (dominant_count / total_valid_scans) * 100

        # 3. Global Report
        logger.info("-" * 60)
        logger.info(f"SCHEMA ANALYSIS RESULT (Optimized)")
        logger.info("-" * 60)
        logger.info(f"Dominant Schema : {len(dominant_signature)} features")
        logger.info(f"Coverage        : {dominant_count}/{total_valid_scans} files ({coverage_percent:.1f}%)")

        # 4. Diagnostic (Deviations)
        if len(schema_map) > 1:
            logger.warning("!" * 60)
            logger.warning("[DIAGNOSTIC] Heterogeneity detected. Analyzing rejections...")

            sorted_schemas = sorted(schema_map.items(), key=lambda item: len(item[1]), reverse=True)

            for signature, filenames in sorted_schemas:
                if signature == dominant_signature:
                    continue

                deviant_set = set(signature)
                missing = dominant_set - deviant_set
                # We also calculate extra features to understand "empty" log groups
                extra = deviant_set - dominant_set

                logger.warning(f"\n>>> DEVIANT GROUP: {len(filenames)} files")
                logger.warning(f"    Examples: {filenames[:3]} ...")

                if missing:
                    logger.warning(f"    CRITICAL MISSING FEATURES: {sorted(list(missing))}")
                elif extra:
                    # If nothing is missing but there is surplus, it is acceptable for the DatasetBuilder!
                    # We flag it just for info.
                    logger.info(f"    (Info) Contains extra features ignored by schema: {sorted(list(extra))}")
                    logger.info(
                        f"    -> These files MIGHT be compatible if included manually, but are strictly different.")

            logger.warning("!" * 60)
        else:
            logger.info("[OK] Perfect homogeneity detected.")

        # 5. Finalize
        # User exclusions are already applied, taking the dominant signature as is.
        config["features"] = sorted(list(dominant_signature))
        logger.info(f"Final Pipeline Configuration: {len(config['features'])} features selected.")

    # 2. Prepare Output Directory
    exp_output_dir = DATASETS_DIR / exp_name

    if exp_output_dir.exists():
        logger.warning(f"Experiment directory '{exp_name}' already exists. Overwriting content.")
        shutil.rmtree(exp_output_dir)

    exp_output_dir.mkdir(parents=True, exist_ok=True)

    # 3. DEFINE SLICING STRATEGY (Controls X Window)
    # We define the physical rules for the input sequence.

    logger.info("Initializing Modular Slicing Strategy for Features (X)...")

    my_slicer = ModularFlightSlicer(
        phases=[8, 9, 10, 11, 12, 13],  # The topological envelope

        # Start Condition: Find index where HEIGHT <= n ft
        start_condition=ThresholdCondition(
            parameter="HEIGHT",
            threshold=800,
            operator='<='
        ),

        end_condition=EventOffsetCondition(
            trigger_phase=12,
            time_offset=5.0
        )
    )

    # 4. Initialize Builder
    builder = DatasetBuilder(input_dir=PROCESSED_DIR, output_dir=exp_output_dir)

    try:
        # 5. Execute Build (Dependency Injection)
        datasets = builder.build_dataset(
            features=config["features"],
            target=config["target"],
            slicer=my_slicer,  # Passing the X strategy
            target_phase=config["target_phase"],
            extraction_method=config.get("extraction_method", "max"),
            test_size=config.get("test_size", 0.2),
            val_size=config.get("val_size", 0.2),
            random_state=config.get("random_state", 42)
        )

        # 6. Final Summary
        X_train = datasets["X_train"]
        logger.info("\n--- GENERATION SUCCESS ---")
        logger.info(f"Dataset location: {exp_output_dir / 'dataset_dense.npz'}")
        logger.info(f"Train set shape:  {X_train.shape} (Samples, Time, Features)")
        logger.info("Ready for training.")

    except Exception as e:
        logger.critical(f"Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()