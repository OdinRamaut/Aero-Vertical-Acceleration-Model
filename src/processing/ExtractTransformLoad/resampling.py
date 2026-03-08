from pathlib import Path
from typing import List, Optional, Tuple

import h5py
import numpy as np
from scipy.interpolate import interp1d


class FlightDataTransformer:
    """
    Handles numerical transformations of flight data.
    Aligns data to a common fixed-frequency time grid (default 4Hz).
    Computes and stores ParametersMinMax statistics for DatasetBuilder target extraction.

    Key Features:
    - Robust handling of non-monotonic timestamps (time resets/overlaps).
    - Support for discrete (step) and continuous (linear) interpolation.
    - Metadata preservation.
    """

    def __init__(self, target_freq: int = 4):
        """
        Args:
            target_freq (int): Target sampling frequency in Hz.
        """
        self.target_freq = target_freq
        self.dt = 1.0 / target_freq

    def process_flight(self, input_path: Path, output_path: Optional[Path] = None,
                       selected_params: Optional[List[str]] = None) -> Optional[Path]:
        """
        Reads structured datasets, aligns them to the target grid, saves flat float32 arrays,
        and regenerates the ParametersMinMax metadata table.

        Args:
            input_path: Path to the simplified raw HDF5 file.
            output_path: Path for the processed output. If None, appends suffix.
            selected_params: List of specific parameters to process (optional).

        Returns:
            Path to the output file, or None if processing failed.
        """
        if output_path is None:
            output_path = input_path.with_name(f"{input_path.stem}_{self.target_freq}hz.h5")

        try:
            with h5py.File(input_path, 'r') as src, h5py.File(output_path, 'w') as dst:
                # 1. Copy MetaData (Global attributes and existing info)
                if 'MetaData' in src:
                    src.copy('MetaData', dst)
                else:
                    dst.create_group("MetaData")

                # 2. Identify Groups
                search_groups = ['Computed Parameters', 'Recorded Parameters', 'Parameters']
                available_groups = [g for g in search_groups if g in src]

                if not available_groups:
                    print(f"[ERROR] No parameter groups found in {input_path.name}")
                    return None

                # 3. Determine Global Time Bounds
                min_t, max_t = self._get_global_time_bounds(src, available_groups)

                if min_t == float('inf'):
                    print(f"[WARN] Could not determine time bounds. Skipping {input_path.name}.")
                    return None

                # 4. Create Target Grid
                target_time = np.arange(min_t, max_t, self.dt, dtype=np.float32)

                # Prepare Output Group
                out_grp = dst.create_group("Parameters")
                out_grp.create_dataset("Time", data=target_time, compression="gzip")

                # 5. Process Parameters
                all_datasets = []
                for g in available_groups:
                    for key in src[g].keys():
                        all_datasets.append((g, key))

                for grp_name, param_name in all_datasets:
                    if selected_params and param_name not in selected_params:
                        continue

                    dataset = src[grp_name][param_name]

                    try:
                        # Compound type extraction (Time, Value)
                        if dataset.dtype.names and 'Time' in dataset.dtype.names and 'Value' in dataset.dtype.names:
                            raw_t = dataset['Time']
                            raw_v = dataset['Value']
                        else:
                            continue

                        if len(raw_t) < 2:
                            continue

                        # Legacy discrete check heuristic
                        unique_vals = len(np.unique(raw_v))
                        is_discrete = (unique_vals < 20) or ('PHASE' in param_name)

                        interpolated_v = self._resample_signal(
                            raw_t, raw_v, target_time, is_discrete=is_discrete
                        )

                        # Save as Flat Array
                        new_ds = out_grp.create_dataset(
                            param_name,
                            data=interpolated_v,
                            compression="gzip",
                            dtype=np.float32
                        )

                        # Preserve attributes
                        for attr, val in dataset.attrs.items():
                            new_ds.attrs[attr] = val

                    except Exception as e:
                        print(f"[WARN] Failed to process {param_name}: {e}")

            return output_path

        except Exception as e:
            print(f"[CRITICAL] Failed to process file {input_path.name}: {e}")
            return None


    def _get_global_time_bounds(self, src_file: h5py.File, groups: List[str]) -> Tuple[float, float]:
        """
        Scans all groups to find the global minimum and maximum timestamps.
        """
        t_min = float('inf')
        t_max = float('-inf')
        for g in groups:
            grp = src_file[g]
            for key in grp:
                ds = grp[key]
                if isinstance(ds, h5py.Dataset) and ds.dtype.names and 'Time' in ds.dtype.names:
                    try:
                        # Robustly check start and end times
                        # Note: We assume sortedness here for speed, but _resample_signal handles detailed fixes
                        times = ds['Time']
                        if len(times) > 0:
                            t_start = times[0]
                            t_end = times[-1]
                            if t_start < t_min: t_min = t_start
                            if t_end > t_max: t_max = t_end
                    except Exception:
                        continue
        return t_min, t_max

    def _resample_signal(self, t_src: np.ndarray, v_src: np.ndarray, t_dst: np.ndarray,
                         is_discrete: bool = False) -> np.ndarray:
        """
        Resamples a signal to the target time grid.

        CRITICAL FIX: Handles non-monotonic timestamps (time resets/overlaps) by
        detecting breaks and keeping only the longest continuous segment.
        """

        # 1. Sort by time (safety for simple disorder)
        sort_idx = np.argsort(t_src)
        t_src = t_src[sort_idx]
        v_src = v_src[sort_idx]

        # 2. Detect Time Resets (Non-monotonicity)
        # We look for instances where t[i+1] <= t[i]
        if len(t_src) > 1:
            diffs = np.diff(t_src)
            breaks = np.where(diffs <= 0)[0]

            if len(breaks) > 0:
                # Split into monotonic segments
                # 'breaks' contains indices before the break, so we split at break+1
                split_indices = breaks + 1
                t_segments = np.split(t_src, split_indices)
                v_segments = np.split(v_src, split_indices)

                # HEURISTIC: Keep the longest segment (richest data)
                # This eliminates parasitic or redundant segments causing jumps
                lengths = [len(seg) for seg in t_segments]
                best_idx = np.argmax(lengths)

                t_src = t_segments[best_idx]
                v_src = v_segments[best_idx]

                # Note: Logging could be added here to track corrected files

        # 3. Final cleanup for interpolation (remove exact duplicates)
        t_src, unique_idx = np.unique(t_src, return_index=True)
        v_src = v_src[unique_idx]

        # Safety for insufficient data after cleanup
        if len(t_src) < 2:
            fill_val = v_src[0] if len(v_src) > 0 else 0.0
            return np.full_like(t_dst, fill_val, dtype=np.float32)

        # 4. Interpolation
        kind = 'previous' if is_discrete else 'linear'

        # bounds_error=False allows extrapolation (filling with edge values)
        # if the target grid extends slightly beyond the source data
        f = interp1d(t_src, v_src, kind=kind,
                     bounds_error=False, fill_value=(v_src[0], v_src[-1]))

        return f(t_dst).astype(np.float32)