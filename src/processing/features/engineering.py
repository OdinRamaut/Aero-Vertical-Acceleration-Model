from pathlib import Path
from typing import List

import h5py
import numpy as np
from scipy.signal import butter, sosfiltfilt
from scipy.ndimage import maximum_filter1d, minimum_filter1d

class FeatureEngineer:
    """
    Enriches the dataset with derived physical parameters.
    Operates on processed (aligned) HDF5 files.
    """

    def __init__(self, sampling_freq: float = 4.0):
        self.fs = sampling_freq

    def process_file(self, file_path: Path) -> None:
        """
        Applies feature engineering in-place to the specified HDF5 file.
        New parameters are added directly to the 'Parameters' group.
        """
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        # Open in 'r+' mode (Read/Write) to append new datasets
        with h5py.File(file_path, 'r+') as f:
            if 'Parameters' not in f: return
            params = f['Parameters']

            if 'Time' not in params: return
            time = params['Time'][:]

            # 1. Compute Derivatives (Dynamics)
            # Added PITCH_C, ROLL_C, IVV_C, RALTC to capture rates
            self._compute_derivatives(params, time,
                                      ["TLA1", "TLA2", "N11", "N12", "PITCH_C", "ROLL_C", "IVV_C", "RALTC"])

            # 1b. Second Derivatives (Accelerations)
            # Creating 'G-Force' equivalent for vertical axis
            if "DELTA_IVV_C" in params:
                self._compute_derivatives(params, time, ["DELTA_IVV_C"])

            # 2. Physics & Energies
            self._compute_combined_wind(params)

            if "GS_C" in params and "IVV_C" in params:
                self._compute_glide_slope(params)

            if "GW_C" in params and "GS_C" in params and "RALTC" in params:
                self._compute_energy_state(params)

            # 3. Deviations (Pilot Logic) <--- NEW STEP
            self._compute_deviations(params)

            # 4. Interactions (Couplings) <--- NEW STEP
            self._compute_interactions(params)

            # 5. Rolling Statistics (Turbulence & History) <--- UPDATED STEP
            # Captures history (Mean/Max) and volatility (Std)
            targets = ["IVV_C", "PITCH_C", "ROLL_C", "GS_C", "N11", "N12"]
            for name in targets:
                if name in params:
                    self._compute_rolling_stats(params, name, window_sec=3.0)

    def _compute_derivatives(self, params: h5py.Group, time: np.ndarray, target_names: List[str]):
        """
        Computes the gradient dx/dt for specified parameters.
        """
        for name in target_names:
            if name in params:
                try:
                    data = params[name][:]
                    # np.gradient computes central difference
                    derivative = np.gradient(data, time)

                    self._save_dataset(params, f"DELTA_{name}", derivative,
                                       attrs={"Description": f"Time derivative of {name}"})
                except Exception as e:
                    print(f"[WARN] Failed derivative for {name}: {e}")

    def _compute_combined_wind(self, params: h5py.Group):
        """
        Combines HEAD_WIND and TAIL_WIND into a single signed vector.
        Convention: Head Wind > 0, Tail Wind < 0.
        """
        # Check naming convention (depends on raw data)
        hw_key = "HEAD_WIND" if "HEAD_WIND" in params else None
        tw_key = "TAIL_WIND" if "TAIL_WIND" in params else None

        if hw_key and tw_key:
            hw = params[hw_key][:]
            tw = params[tw_key][:]

            # Logic: A plane usually has either Head or Tail wind component active
            # If both exist (raw data often splits them), we combine.
            combined_wind = hw - tw

            self._save_dataset(params, "HEAD_TAIL_WIND", combined_wind,
                               attrs={"Description": "Combined Head(+) and Tail(-) wind", "Unit": "kt"})

    def _compute_glide_slope(self, params: h5py.Group):
        """
        Calculates Glide Slope angle in degrees.
        Includes Low-Pass filtering to stabilize the noisy geometric calculation.
        """
        try:
            gs_knots = params["GS_C"][:]  # Ground Speed (kt)
            ivv_fpm = params["IVV_C"][:]  # Vertical Speed (ft/min)

            # Conversion: 1 kt = 101.2686 ft/min
            gs_fpm = gs_knots * 101.2686

            # Avoid division by zero (on ground, speed ~ 0)
            # We clip GS to a small epsilon
            gs_fpm = np.maximum(gs_fpm, 0.1)

            # 1. Pre-filtering (0.5 Hz)
            # Essential because derivative-like calculations amplify high-freq noise
            gs_smooth = self._butter_lowpass_filter(gs_fpm, cutoff=0.5, fs=self.fs, order=1)
            ivv_smooth = self._butter_lowpass_filter(ivv_fpm, cutoff=0.5, fs=self.fs, order=1)

            # 2. Calculation: gamma = atan(Vz / Vx)
            # Standard aviation sign convention: Flight Path Angle is positive when climbing
            # But Glide Slope is often viewed as positive for descent in some contexts.
            # Let's keep standard physics: Climb > 0, Descent < 0.
            slope_rad = np.arctan2(ivv_smooth, gs_smooth)
            slope_deg = np.degrees(slope_rad)

            # 3. Post-filtering (1.0 Hz) for clean signal
            final_slope = self._butter_lowpass_filter(slope_deg, cutoff=1.0, fs=self.fs, order=1)

            # Save Derived Features
            self._save_dataset(params, "GLD_SLOPE", final_slope, attrs={"Unit": "deg"})
            self._save_dataset(params, "GS_C_SMOOTH", gs_smooth / 101.2686, attrs={"Unit": "kt"})
            self._save_dataset(params, "IVV_C_SMOOTH", ivv_smooth, attrs={"Unit": "ft/min"})

            # Compute deviation from standard -3.0 degree path
            # Positive = High/Shallow, Negative = Low/Steep
            gld_dev = final_slope - (-3.0)
            self._save_dataset(params, "GLD_DEVIATION", gld_dev,
                               attrs={"Description": "Deviation from -3 deg glide slope"})

        except Exception as e:
            print(f"[WARN] Failed to compute GLD_SLOPE: {e}")

    def _compute_energy_state(self, params: h5py.Group):
        """
        Computes kinetic and potential energy indicators.
        Note: Units are not strict SI, but relative magnitude matters for ML.
        """
        try:
            # Gross Weight (kg), Ground Speed (kt), Radio Altitude (ft)
            gw = params["GW_C"][:]
            gs = params["GS_C"][:]
            h = params["RALTC"][:]

            # Approximate conversions for physical consistency
            gs_ms = gs * 0.5144  # kt to m/s
            h_m = h * 0.3048  # ft to m
            g = 9.81

            # Kinetic Energy: 1/2 * m * v^2
            # We normalize by 1e6 to have readable numbers (MegaJoules approx)
            ek = 0.5 * gw * (gs_ms ** 2) / 1e6

            # Potential Energy: m * g * h
            ep = gw * g * h_m / 1e6

            # Total Energy
            etot = ek + ep

            self._save_dataset(params, "ENERGY_KINETIC", ek,
                               attrs={"Description": "Kinetic Energy Proxy", "Unit": "MJ"})
            self._save_dataset(params, "ENERGY_POTENTIAL", ep,
                               attrs={"Description": "Potential Energy Proxy", "Unit": "MJ"})
            self._save_dataset(params, "ENERGY_TOTAL", etot, attrs={"Description": "Total Energy Proxy", "Unit": "MJ"})

        except Exception as e:
            print(f"[WARN] Failed energy compute: {e}")

    def _compute_deviations(self, params: h5py.Group):
        """
        Computes deviations from pilot targets (Speed, Path).
        """
        try:
            # Speed Deviation: Difference between Indicated Airspeed and Target Approach Speed
            if "IAS_C" in params and "VAPP" in params:
                ias = params["IAS_C"][:]
                vapp = params["VAPP"][:]

                # Handle scalar or vector VAPP
                speed_dev = ias - vapp
                self._save_dataset(params, "SPEED_DEVIATION", speed_dev,
                                   attrs={"Description": "IAS - VAPP (Speed Error)", "Unit": "kt"})
        except Exception as e:
            print(f"[WARN] Failed deviation compute: {e}")

    def _compute_interactions(self, params: h5py.Group):
        """
        Computes physical coupling between variables.
        """
        try:
            # Energy-Pitch Index: High Pitch * High Power = High Energy State
            if "PITCH_C" in params and "N11" in params:
                pitch = params["PITCH_C"][:]
                n1 = params["N11"][:]

                # Interaction term
                energy_idx = pitch * n1
                self._save_dataset(params, "PITCH_POWER_IDX", energy_idx,
                                   attrs={"Description": "Pitch * N1 interaction"})
        except Exception as e:
            print(f"[WARN] Failed interaction compute: {e}")

    def _compute_rolling_stats(self, params: h5py.Group, feature_name: str, window_sec: float):
        """
        Computes Rolling Mean, Std, Max, and Min.
        Replaces the old _compute_rolling_stability method.
        """
        try:
            data = params[feature_name][:]

            # Window setup
            window_size = int(window_sec * self.fs)
            if window_size < 2: window_size = 2

            # 1. Rolling Max/Min (Efficient implementation)
            rolling_max = maximum_filter1d(data, size=window_size)
            rolling_min = minimum_filter1d(data, size=window_size)

            self._save_dataset(params, f"{feature_name}_MAX_{int(window_sec)}s", rolling_max)
            self._save_dataset(params, f"{feature_name}_MIN_{int(window_sec)}s", rolling_min)

            # 2. Rolling Mean & Std
            kernel = np.ones(window_size) / window_size
            rolling_mean = np.convolve(data, kernel, mode='same')

            # Std Dev = sqrt(E[x^2] - (E[x])^2)
            mean_sq = np.convolve(data ** 2, kernel, mode='same')
            var = np.maximum(mean_sq - rolling_mean ** 2, 0)
            std = np.sqrt(var)

            self._save_dataset(params, f"{feature_name}_MEAN_{int(window_sec)}s", rolling_mean)
            self._save_dataset(params, f"{feature_name}_STD_{int(window_sec)}s", std)

        except Exception as e:
            print(f"[WARN] Failed rolling stats for {feature_name}: {e}")

    def _save_dataset(self, group: h5py.Group, name: str, data: np.ndarray, attrs: dict = None):
        """Helper to overwrite/create dataset safely."""
        if name in group:
            del group[name]

        dset = group.create_dataset(name, data=data, dtype=np.float32, compression="gzip")
        if attrs:
            for k, v in attrs.items():
                dset.attrs[k] = v

    @staticmethod
    def _butter_lowpass_filter(data: np.ndarray, cutoff: float, fs: float, order: int = 1) -> np.ndarray:
        nyq = 0.5 * fs
        if cutoff >= nyq:
            return data  # No filtering if cutoff is too high
        normal_cutoff = cutoff / nyq
        sos = butter(order, normal_cutoff, btype='low', analog=False, output='sos')
        return sosfiltfilt(sos, data)