from abc import ABC, abstractmethod
from typing import Optional, Dict

import numpy as np


class TargetExtractor(ABC):
    """
    Abstract Base Class defining the contract for target scalar extraction.
    Represents the operator E such that y = E(S_Phi), where S_Phi is the signal
    restricted to the flight phase Phi.
    """

    @abstractmethod
    def extract(self, signal: np.ndarray, time_vector: Optional[np.ndarray] = None) -> float:
        """
        Extracts a scalar feature from a time-series segment.

        Args:
            signal (np.ndarray): 1D array containing the signal values on the specific phase.
            time_vector (Optional[np.ndarray]): 1D array containing time steps.
                                                Required for integral-based extractors.

        Returns:
            float: The extracted scalar target y.
        """
        pass


class MaxPeakExtractor(TargetExtractor):
    """
    Extracts the maximum peak value of the signal.
    y = max(S_Phi)

    Use case: Structural limit loads, maximum vertical acceleration (VRTG), peak temperature.
    """

    def extract(self, signal: np.ndarray, time_vector: Optional[np.ndarray] = None) -> float:
        if len(signal) == 0:
            return 0.0
        return float(np.max(signal))


class LastValueExtractor(TargetExtractor):
    """
    Extracts the final value of the sequence.
    y = s_T, where T is the last time step of the phase.

    Use case: Fuel remaining, cumulative fatigue index at end of flight, final error.
    """

    def extract(self, signal: np.ndarray, time_vector: Optional[np.ndarray] = None) -> float:
        if len(signal) == 0:
            return 0.0
        return float(signal[-1])


class MeanExtractor(TargetExtractor):
    """
    Computes the arithmetic mean of the signal.
    y = (1/T) * sum(s_t)

    Use case: Average Exhaust Gas Temperature (EGT), mean vibration level.
    """

    def extract(self, signal: np.ndarray, time_vector: Optional[np.ndarray] = None) -> float:
        if len(signal) == 0:
            return 0.0
        return float(np.mean(signal))


class IntegralExtractor(TargetExtractor):
    """
    Computes the time integral of the signal using the Trapezoidal rule.
    y = integral(s(t) dt) over Phi.

    Use case: Total energy dissipated, cumulative dose.
    Requires: 'time_vector' to be provided.
    """

    def extract(self, signal: np.ndarray, time_vector: Optional[np.ndarray] = None) -> float:
        if len(signal) < 2 or time_vector is None:
            return 0.0

        # Ensure dimensions match
        if len(time_vector) != len(signal):
            raise ValueError("Signal and Time vector must have the same length for integration.")

        return float(np.trapz(signal, x=time_vector))


# =============================================================================
# REGISTRY & FACTORY
# =============================================================================

# Registry of available strategies
# Keys correspond to the 'extraction_method' field in the experiment config.
_EXTRACTORS: Dict[str, TargetExtractor] = {
    "max": MaxPeakExtractor(),
    "last": LastValueExtractor(),
    "mean": MeanExtractor(),
    "integral": IntegralExtractor()
}


def get_extractor(method_name: str) -> TargetExtractor:
    """
    Factory function to retrieve a target extraction strategy.

    Args:
        method_name (str): The identifier of the method (e.g., 'max', 'last').

    Returns:
        TargetExtractor: An instance implementing the requested strategy.

    Raises:
        ValueError: If the method_name is unknown.
    """
    if method_name not in _EXTRACTORS:
        raise ValueError(f"Unknown extraction method: '{method_name}'. "
                         f"Available: {list(_EXTRACTORS.keys())}")
    return _EXTRACTORS[method_name]