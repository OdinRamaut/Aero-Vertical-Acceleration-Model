import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Dict, Optional

import numpy as np

# Setup logger for this module
logger = logging.getLogger(__name__)


# =============================================================================
# 1. DATA STRUCTURES
# =============================================================================

@dataclass
class SliceResult:
    """
    Encapsulates the final result of a slicing operation on a flight.
    """
    start_idx: int
    end_idx: int
    is_valid: bool
    rejection_reason: Optional[str] = None


@dataclass
class ConditionResult:
    """
    Encapsulates the result of a single condition check (e.g., finding a threshold).
    """
    index: int  # The calculated index (start or end candidate)
    found: bool  # Whether the condition was met
    reason: Optional[str] = None


# =============================================================================
# 2. ABSTRACT INTERFACES
# =============================================================================

class FlightSlicer(ABC):
    """
    Abstract Strategy for slicing flight data.
    The DatasetBuilder relies on this contract.
    """

    @abstractmethod
    def slice(self, flight_data: Dict[str, np.ndarray]) -> SliceResult:
        """
        Determines the valid time interval [start, end].

        Args:
            flight_data: Dictionary containing full numpy arrays of flight parameters.
                         Must include 'Time', 'FLIGHT_PHASE', and any parameter
                         required by the specific conditions.
        """
        pass


class SlicingCondition(ABC):
    """
    Interface for a specific trimming rule (Condition).
    Used by ModularFlightSlicer via composition.
    """

    @abstractmethod
    def find_index(self, flight_data: Dict[str, np.ndarray], reference_idx: int = 0) -> ConditionResult:
        """
        Determines a cutoff index based on the condition logic.

        Args:
            flight_data: Full flight data context.
            reference_idx: A reference index to optimize search or define direction.
                           - For Start conditions: typically the start of the phase envelope.
                           - For End conditions: typically the end of the phase envelope.
        """
        pass

    @property
    @abstractmethod
    def required_features(self) -> List[str]:
        """
        Declares which HDF5 parameters are needed for this condition
        (e.g., ['HEIGHT'] or ['IVV_C']).
        """
        pass


# =============================================================================
# 3. CONCRETE CONDITIONS (The "Bricks")
# =============================================================================

class ThresholdCondition(SlicingCondition):
    """
    Finds the first index where a parameter meets a threshold comparison.
    Useful for: "Start when HEIGHT <= 300" or "Start when VAPP > 100".
    """

    def __init__(self, parameter: str, threshold: float, operator: str = '<='):
        """
        Args:
            parameter: Name of the parameter (e.g., 'HEIGHT').
            threshold: Value to compare against.
            operator: Comparison operator ('<=', '>=', '<', '>').
        """
        self.parameter = parameter
        self.threshold = threshold
        self.operator = operator

        valid_ops = {'<=', '>=', '<', '>'}
        if operator not in valid_ops:
            raise ValueError(f"Invalid operator '{operator}'. Must be one of {valid_ops}")

    def find_index(self, flight_data: Dict[str, np.ndarray], reference_idx: int = 0) -> ConditionResult:
        if self.parameter not in flight_data:
            return ConditionResult(0, False, f"Missing parameter: {self.parameter}")

        data = flight_data[self.parameter]

        # We search strictly AFTER (or at) the reference index.
        # This prevents finding a valid height that happened hours before the landing phase.
        if reference_idx >= len(data):
            return ConditionResult(0, False, "Reference index out of bounds")

        sub_data = data[reference_idx:]

        # Apply the operator logic
        if self.operator == '<=':
            mask = sub_data <= self.threshold
        elif self.operator == '>=':
            mask = sub_data >= self.threshold
        elif self.operator == '<':
            mask = sub_data < self.threshold
        elif self.operator == '>':
            mask = sub_data > self.threshold
        else:
            return ConditionResult(0, False, f"Unknown operator {self.operator}")

        # Check if condition is ever met in the valid segment
        if not np.any(mask):
            return ConditionResult(0, False,
                                   f"{self.parameter} never meets {self.operator} {self.threshold} after ref index")

        # np.argmax on a boolean array returns the index of the first True
        offset = np.argmax(mask)
        absolute_index = reference_idx + offset

        return ConditionResult(absolute_index, True)

    @property
    def required_features(self) -> List[str]:
        return [self.parameter]


class EventOffsetCondition(SlicingCondition):
    """
    Defines a cutoff based on a time offset from a specific event (Flight Phase trigger).
    Useful for: "Stop 100s before Phase 13 starts".
    """

    def __init__(self, trigger_phase: int, time_offset: float):
        """
        Args:
            trigger_phase: The phase ID to look for (e.g., 13 for Touchdown/Taxi).
            time_offset: Seconds to add to the trigger time.
                         Negative value = Before the event.
                         Positive value = After the event.
        """
        self.trigger_phase = trigger_phase
        self.time_offset = time_offset

    def find_index(self, flight_data: Dict[str, np.ndarray], reference_idx: int = 0) -> ConditionResult:
        # Check dependencies
        if 'FLIGHT_PHASE' not in flight_data or 'Time' not in flight_data:
            return ConditionResult(0, False, "Missing FLIGHT_PHASE or Time")

        phases = flight_data['FLIGHT_PHASE']
        time = flight_data['Time']

        # 1. Find the event in the WHOLE flight
        # (We usually search globally because the event (e.g. Taxi) might be
        # outside the initial envelope defined by Approach phases).
        trigger_indices = np.where(phases == self.trigger_phase)[0]

        if trigger_indices.size == 0:
            return ConditionResult(0, False, f"Trigger phase {self.trigger_phase} not found in flight")

        # We take the first occurrence of the trigger phase
        trigger_idx = trigger_indices[0]
        trigger_time = time[trigger_idx]

        # 2. Calculate target time
        target_time = trigger_time + self.time_offset

        # 3. Find the index corresponding to this time
        # searchsorted finds the insertion point to maintain order.
        # side='right' - 1 gives the index where time <= target_time
        cut_idx = np.searchsorted(time, target_time, side='right') - 1

        # Boundary safety
        cut_idx = max(0, min(cut_idx, len(time) - 1))

        return ConditionResult(cut_idx, True)

    @property
    def required_features(self) -> List[str]:
        return ['FLIGHT_PHASE', 'Time']


# =============================================================================
# 4. ORCHESTRATOR (The Strategy Implementation)
# =============================================================================

class ModularFlightSlicer(FlightSlicer):
    """
    Orchestrates the slicing process using composable conditions.

    Sequence:
    1. Phase Envelope: Determines the base window (longest continuous segment of allowed phases).
    2. Start Trimming: Moves start forward using start_condition (if provided).
    3. End Trimming: Moves end backward using end_condition (if provided).
    """

    def __init__(self,
                 phases: List[int],
                 start_condition: Optional[SlicingCondition] = None,
                 end_condition: Optional[SlicingCondition] = None):
        """
        Args:
            phases: List of valid phases defining the topological envelope.
            start_condition: Rule to trim the beginning (e.g., ThresholdCondition).
            end_condition: Rule to trim the end (e.g., EventOffsetCondition).
        """
        self.phases = phases
        self.start_condition = start_condition
        self.end_condition = end_condition

    def slice(self, flight_data: Dict[str, np.ndarray]) -> SliceResult:
        # --- 0. Global Dependency Check ---
        if 'FLIGHT_PHASE' not in flight_data:
            return SliceResult(0, 0, False, "FLIGHT_PHASE parameter missing")

        phases = flight_data['FLIGHT_PHASE']

        # --- 1. Phase Envelope (Topological Selection) ---
        allowed_mask = np.isin(phases, self.phases)
        allowed_indices = np.where(allowed_mask)[0]

        if allowed_indices.size == 0:
            return SliceResult(0, 0, False, f"No data found for phases {self.phases}")

        # Identify continuous segments
        breaks = np.where(np.diff(allowed_indices) != 1)[0]
        segments = np.split(allowed_indices, breaks + 1)

        # Selection Heuristic:
        # Prioritize the segment with the most distinct phases (richness).
        # If equal, take the longest one.
        best_seg = max(segments, key=lambda s: (len(np.unique(phases[s])), len(s)))

        start_idx = best_seg[0]
        end_idx = best_seg[-1]

        # --- 2. Start Trimming ---
        if self.start_condition:
            # We look for the condition starting from the envelope's start
            res = self.start_condition.find_index(flight_data, reference_idx=start_idx)

            if not res.found:
                return SliceResult(0, 0, False, f"Start Condition failed: {res.reason}")

            # Logic: We can only TIGHTEN the window (move start forward)
            # If the found index is before the current start, we ignore it (or treat as valid depending on semantics).
            # Here, we strictly enforce the condition must be met validly.

            new_start = res.index

            # Safety: Ensure we don't start after the envelope ends (though Step 4 catches this too)
            if new_start > end_idx:
                return SliceResult(0, 0, False, "Start condition met after phase envelope end")

            start_idx = max(start_idx, new_start)

        # --- 3. End Trimming ---
        if self.end_condition:
            # We look for the condition (often global search, so reference_idx might be ignored by implementation
            # or used for optimization). We pass end_idx as context.
            res = self.end_condition.find_index(flight_data,
                                                reference_idx=start_idx
                                                )

            if not res.found:
                return SliceResult(0, 0, False, f"End Condition failed: {res.reason}")

            new_end = res.index

            # Logic: We can only TIGHTEN the window (move end backward)
            end_idx = min(end_idx, new_end)

        # --- 4. Final Validity Check ---
        if start_idx >= end_idx:
            return SliceResult(0, 0, False,
                               f"Resulting window is empty or inverted (Start: {start_idx}, End: {end_idx})")

        return SliceResult(start_idx, end_idx, True, None)