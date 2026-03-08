import sys
from pathlib import Path

import numpy as np

# --- SETUP PATHS ---
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.config import DATASETS_DIR, PADDING_VALUE

# --- CONFIGURATION ---
EXPERIMENT_NAME = "Exp_06_Landing_VRTG_Padding_updated"
FILE_PATH = DATASETS_DIR / EXPERIMENT_NAME / "dataset_dense.npz"


def main():
    print(f"Checking: {FILE_PATH}")

    if not FILE_PATH.exists():
        print("❌ ERROR: File not found.")
        return

    data = np.load(FILE_PATH, allow_pickle=True)

    # 1. Check Shapes
    X_train = data['X_train']
    mask_train = data['mask_train']
    print(f"\n[1] SHAPES:")
    print(f"   X_train: {X_train.shape}")
    print(f"   mask_train: {mask_train.shape}")

    # 2. Check Padding Value
    print(f"\n[2] PADDING CHECK (Value: {PADDING_VALUE}):")

    # Count how many occurrences of the exact padding value exist
    # Using isclose because float equality can be tricky, though -1e9 is integer-like
    is_padded = np.isclose(X_train, PADDING_VALUE)
    pad_count = np.sum(is_padded)
    total_cells = X_train.size

    print(f"   Total cells: {total_cells}")
    print(f"   Padded cells found: {pad_count} ({pad_count / total_cells:.1%})")

    if pad_count == 0:
        print("⚠️ WARNING: No padding values found! Did you regenerate the dataset?")
    else:
        print("✅ Padding values detected.")

    # 3. Check Mask Consistency
    print(f"\n[3] MASK CONSISTENCY:")
    # Logic: If mask is 0 (hidden), value MUST be PADDING_VALUE
    # We check the first feature layer (usually consistent across time)

    # Locations where mask says "Hide" (0)
    masked_indices = (mask_train == 0)

    # Values at these locations in X (first feature)
    values_at_masked = X_train[:, :, 0][masked_indices]

    # Are they all equal to PADDING_VALUE?
    all_correct = np.all(np.isclose(values_at_masked, PADDING_VALUE))

    if all_correct:
        print("✅ Mask perfectly aligns with Padding Value.")
    else:
        print("❌ ERROR: Mismatch! Some masked values are NOT equal to padding value.")
        # Debug info
        sample_errors = values_at_masked[~np.isclose(values_at_masked, PADDING_VALUE)]
        if len(sample_errors) > 0:
            print(f"   Sample incorrect values: {sample_errors[:5]}")

    # 4. Quick Sample
    print(f"\n[4] VISUAL SAMPLE (Flight #0, Feature #0 - Last 10 steps):")
    sample_seq = X_train[0, -10:, 0]
    print(f"   Values: {sample_seq}")


if __name__ == "__main__":
    main()