import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union

import h5py
import numpy as np
from sklearn.model_selection import train_test_split

from src.config import PADDING_VALUE
from src.processing.building.slicing import FlightSlicer
# Internal imports
from src.processing.building.targets import get_extractor

# Setup logger for this module
logger = logging.getLogger(__name__)


class DatasetBuilder:
    """
    Converts processed HDF5 flight files into Machine Learning datasets (X, y, ids).
    Architecture: 'Select & Filter'.
    It takes a list of desired features and builds a dense dataset.
    If a file misses a required feature, it is skipped for this specific dataset build.
    """

    def __init__(self, input_dir: Path, output_dir: Path):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def build_dataset(self,
                      features: List[str],
                      target: str,
                      slicer: FlightSlicer,
                      target_phase: Optional[Union[int, List[int], str]] = None,
                      extraction_method: str = "max",
                      test_size: float = 0.2,
                      val_size: float = 0.2,
                      random_state: int = 42) -> Dict[str, np.ndarray]:
        """
        Orchestrates the construction of the dataset.

        Args:
            features: List of parameter names to use as predictors (X).
            target: Parameter name to use as the target (y).
            slicer: Strategy object to determine valid time windows for X.
            target_phase: Controls where to extract the target (y).
                          - None: Extract from the same window as X (Coupled).
                          - "FULL": Extract from the entire flight signal.
                          - int or List[int]: Extract from specific flight phase(s).
            extraction_method: Strategy to extract scalar y from the time series ('max', 'mean', etc.).
            test_size: Proportion of the dataset to include in the test split.
            val_size: Proportion of the dataset to include in the validation split.
            random_state: Seed for reproducibility.

        Returns:
            Dict containing the dense tensors for train, val, and test sets.
        """
        # Format for logging
        t_phase_log = target_phase
        if target_phase is None:
            t_phase_log = "Coupled with X"

        logger.info(f"Building dataset from {self.input_dir}")
        logger.info(f"Target: {target} | Scope: {t_phase_log} | Extraction Method: {extraction_method}")
        logger.info(f"Features: {len(features)} variables requested.")

        # Gather all HDF5 files
        files = list(self.input_dir.glob("*.h5")) + list(self.input_dir.glob("*.hdf5"))
        if not files:
            raise FileNotFoundError(f"No files found in {self.input_dir}")

        X_list = []
        y_list = []
        ids_list = []
        skipped_count = 0
        success_count = 0

        # 1. Extraction Loop
        for i, fpath in enumerate(files, 1):
            X_seq, y_val = self._extract_flight_data(
                fpath, features, target, slicer, extraction_method, target_phase
            )

            if X_seq is not None and X_seq.shape[0] > 0 and y_val is not None:
                X_list.append(X_seq)
                y_list.append(y_val)
                # Store filename (without extension) as unique ID
                ids_list.append(fpath.stem)
                success_count += 1
                logger.debug(f"[OK] {fpath.name} - Length: {X_seq.shape[0]} steps - Target: {y_val:.4f}")
            else:
                skipped_count += 1
                # Detailed skip reason is logged within _extract_flight_data at DEBUG level

        n_samples = len(X_list)
        logger.info(f"Summary: {n_samples} valid sequences kept. {skipped_count} files skipped.")

        if n_samples == 0:
            raise ValueError("No valid data extracted. See logs/dataset_generation.log for details.")

        # 2. Dense Tensor Construction (Padding)
        logger.info("Constructing dense tensors (Padding)...")
        max_len = max(x.shape[0] for x in X_list)
        n_features = len(features)

        # Initialize dense arrays
        X_dense = np.full((n_samples, max_len, n_features), PADDING_VALUE, dtype=np.float32)
        mask_dense = np.zeros((n_samples, max_len), dtype=np.uint8)
        y_dense = np.array(y_list, dtype=np.float32)
        # Store IDs as fixed-length strings (S64 is usually sufficient)
        ids_dense = np.array(ids_list, dtype='S64')

        # Fill arrays
        for i, x in enumerate(X_list):
            seq_len = x.shape[0]
            X_dense[i, :seq_len, :] = x
            mask_dense[i, :seq_len] = 1

        # 3. Stratified Split
        # We split indices instead of data directly to handle multiple arrays (X, y, mask, ids)
        indices = np.arange(n_samples)

        # Safety check for very small datasets
        if n_samples < 3:
            logger.warning(f"N={n_samples} is too small for splitting. All data goes to TRAIN.")
            train_idx = indices
            val_idx = np.array([], dtype=int)
            test_idx = np.array([], dtype=int)
        else:
            try:
                # First split: Train vs (Val + Test)
                train_idx, temp_idx = train_test_split(indices, test_size=test_size, random_state=random_state)

                # Second split: Val vs Test
                if len(temp_idx) < 2:
                    # Not enough samples left for a second split
                    val_idx = temp_idx
                    test_idx = np.array([], dtype=int)
                else:
                    val_idx, test_idx = train_test_split(temp_idx, test_size=val_size, random_state=random_state)
            except ValueError:
                # Fallback in case of splitting errors
                train_idx = indices
                val_idx = np.array([], dtype=int)
                test_idx = np.array([], dtype=int)

        # 4. Packaging
        datasets = {
            "X_train": X_dense[train_idx], "mask_train": mask_dense[train_idx], "y_train": y_dense[train_idx],
            "ids_train": ids_dense[train_idx],
            "X_val": X_dense[val_idx], "mask_val": mask_dense[val_idx], "y_val": y_dense[val_idx],
            "ids_val": ids_dense[val_idx],
            "X_test": X_dense[test_idx], "mask_test": mask_dense[test_idx], "y_test": y_dense[test_idx],
            "ids_test": ids_dense[test_idx]
        }

        # Metadata for traceability
        save_path = self.output_dir / "dataset_dense.npz"
        np.savez_compressed(
            save_path,
            **datasets,
            feature_names=np.array(features, dtype='S'),
            target_name=np.array([target], dtype='S')
        )
        logger.info(f"Dataset saved to {save_path}")
        return datasets

    def _extract_flight_data(self,
                             file_path: Path,
                             features: List[str],
                             target: str,
                             slicer: FlightSlicer,
                             extraction_method: str,
                             target_phase: Optional[Union[int, List[int], str]]
                             ) -> Tuple[Optional[np.ndarray], Optional[float]]:
        """
            Orchestrates the extraction of the feature sequence (X) and the scalar target (y) from a raw HDF5 flight file.
            This method enforces data integrity (presence of features) and applies the temporal slicing strategy.

            Args:
                file_path (Path): Absolute path to the source HDF5 file.
                features (List[str]): List of parameter names to be stacked as the input tensor X.
                target (str): Name of the parameter to be used as the target variable y.
                slicer (FlightSlicer): The slicing strategy instance responsible for determining the valid time window
                                       for X (e.g., based on phase envelope and physical thresholds).
                extraction_method (str): The reduction strategy to transform the target time series into a scalar y.
                                         Supported values: 'max', 'mean', 'last', 'integral'.
                target_phase (Optional[Union[int, List[int], str]]): Defines the temporal scope for target extraction,
                                                                     allowing for X-y decoupling (Forecasting vs Diagnosis).
                    - None (Default): Coupled mode. y is extracted from the exact same time window as X.
                    - "FULL": Global mode. y is extracted from the entire flight duration.
                    - int or List[int]: Phase-specific mode. y is extracted strictly from the specified flight phase(s)
                                        (e.g., 13 for Touchdown, or [12, 13] for Flare + Taxi).

            Returns:
                Tuple[Optional[np.ndarray], Optional[float]]:
                    - X_seq: The dense feature tensor of shape (Time, n_features).
                    - y_val: The extracted scalar target.
                    Returns (None, None) if the file is invalid, missing features, or if the slicing/target scope is empty.
            """
        try:
            with h5py.File(file_path, 'r') as f:
                if 'Parameters' not in f:
                    logger.debug(f"[SKIP] {file_path.name}: Group 'Parameters' not found.")
                    return None, None
                params = f['Parameters']

                # --- 1. Context Loading for Slicing (Dynamic) ---
                # Determine what the slicer needs to perform its job.
                # 'Time' and 'FLIGHT_PHASE' are considered mandatory base context.
                context_keys = {"Time", "FLIGHT_PHASE"}

                # Dynamic introspection of the Slicer strategy
                if hasattr(slicer, 'start_condition') and slicer.start_condition:
                    context_keys.update(slicer.start_condition.required_features)
                if hasattr(slicer, 'end_condition') and slicer.end_condition:
                    context_keys.update(slicer.end_condition.required_features)

                flight_context = {}
                missing_context = []
                for k in context_keys:
                    if k in params:
                        flight_context[k] = params[k][:]
                    else:
                        missing_context.append(k)

                if missing_context:
                    # We log this but allow the slicer to fail gracefully if it wants to
                    logger.debug(f"Context missing keys {missing_context} for {file_path.name}")

                # --- 2. Execute Slicing Strategy (For X) ---
                slice_result = slicer.slice(flight_context)

                if not slice_result.is_valid:
                    logger.debug(f"[SKIP] {file_path.name}: {slice_result.rejection_reason}")
                    return None, None

                # --- 3. Extract X (Features) ---
                x_indices = np.arange(slice_result.start_idx, slice_result.end_idx + 1)

                # Verify features existence
                missing_features = [feat for feat in features if feat not in params]
                if missing_features:
                    logger.debug(f"[SKIP] {file_path.name}: Missing features {missing_features}")
                    return None, None

                x_data = []
                for feat in features:
                    val = params[feat][:]
                    # Safety check on length
                    if len(val) <= x_indices[-1]:
                        logger.warning(f"[SKIP] {file_path.name}: Feature {feat} length mismatch.")
                        return None, None
                    x_data.append(val[x_indices])

                # Stack features: (Time, Features)
                X_seq = np.column_stack(x_data)

                # --- 4. Extract y (Target) ---
                if target not in params:
                    logger.debug(f"[SKIP] {file_path.name}: Missing target {target}")
                    return None, None

                target_full_signal = params[target][:]

                # --- DECOUPLED LOGIC for Target Extraction ---

                # CASE 1: "FULL" (Entire Flight)
                if target_phase == "FULL":
                    y_signal_segment = target_full_signal[:]

                    time_vector_segment = None
                    if extraction_method == "integral":
                        if "Time" in params:
                            time_vector_segment = params["Time"][:]

                # CASE 2: List of phases or Single Phase (e.g., [12, 13] or 13)
                elif target_phase is not None:
                    if "FLIGHT_PHASE" not in params:
                        logger.debug(f"[SKIP] {file_path.name}: FLIGHT_PHASE missing for target.")
                        return None, None

                    phases_full = params["FLIGHT_PHASE"][:]

                    # Normalize to list for np.isin
                    if isinstance(target_phase, int):
                        phases_to_keep = [target_phase]
                    else:
                        phases_to_keep = target_phase  # It is already a list

                    # Create Mask
                    target_mask = np.isin(phases_full, phases_to_keep)

                    if not np.any(target_mask):
                        # The target phase does not exist in this flight
                        logger.debug(f"[SKIP] {file_path.name}: Target phases {phases_to_keep} not found.")
                        return None, None

                    y_signal_segment = target_full_signal[target_mask]

                    time_vector_segment = None
                    if extraction_method == "integral":
                        if "Time" in params:
                            time_vector_segment = params["Time"][:][target_mask]

                # CASE 3: Default (None) -> Coupled with X
                else:
                    y_signal_segment = target_full_signal[x_indices]

                    time_vector_segment = None
                    if extraction_method == "integral":
                        if "Time" in flight_context:
                            time_vector_segment = flight_context["Time"][x_indices]
                        elif "Time" in params:
                            time_vector_segment = params["Time"][:][x_indices]

                # Delegate scalar calculation
                extractor = get_extractor(extraction_method)
                y_scalar = extractor.extract(y_signal_segment, time_vector_segment)

                return X_seq, y_scalar

        except Exception as e:
            logger.error(f"[ERR] Read error on {file_path.name}: {e}")
            return None, None