import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# --- CONFIGURATION ---
EXPERIMENT_NAME = "Exp_05_Landing_VRTG_Analysis_Prediction_ALL_FEATURES_touch_included"

# Setup Paths
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent
if str(project_root) not in sys.path: sys.path.append(str(project_root))
try:
    from src.config import DATASETS_DIR
except ImportError:
    DATASETS_DIR = Path("C:/Users/odino/PycharmProjects/AI_ALPHA_2/data/output/datasets")

DATASET_PATH = DATASETS_DIR / EXPERIMENT_NAME / "dataset_dense.npz"

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def clean_feature_name(f):
    """Décode proprement les noms de features (bytes ou str)."""
    if isinstance(f, bytes):
        return f.decode('utf-8')
    s = str(f)
    # Si la conversion str() a gardé le format "b'Name'", on nettoie
    if s.startswith("b'") and s.endswith("'"):
        return s[2:-1]
    return s


def main():
    logger.info("--- DATA FORENSICS: ALIGNMENT & PHYSICS CHECK (PADDING AWARE) ---")

    if not DATASET_PATH.exists():
        logger.error(f"Dataset not found at {DATASET_PATH}")
        return

    data = np.load(DATASET_PATH, allow_pickle=True)
    X = data['X_train']
    y = data['y_train']
    mask = data['mask_train'].astype(bool) if 'mask_train' in data else None

    # --- CORRECTION DU DECODAGE ---
    feat_names = [clean_feature_name(f) for f in data['feature_names']]
    logger.info(f"Loaded Shape: {X.shape}")
    logger.info(f"Features detected: {feat_names}")

    try:
        # 1. IDENTIFY KEY SENSORS
        # Adaptation : On utilise HEIGHT si RALTC n'est pas disponible
        idx_alt = -1
        if 'RALTC' in feat_names:
            idx_alt = feat_names.index('RALTC')
            logger.info("Sensor found: RALTC (Radio Altitude)")
        elif 'HEIGHT' in feat_names:
            idx_alt = feat_names.index('HEIGHT')
            logger.info("Sensor found: HEIGHT (used as Altitude)")
        else:
            raise ValueError("Ni RALTC, ni HEIGHT n'ont été trouvés.")

        if 'IVV_C' in feat_names:
            idx_vs = feat_names.index('IVV_C')
            logger.info("Sensor found: IVV_C (Vertical Speed)")
        else:
            raise ValueError("IVV_C (Vertical Speed) introuvable.")

    except ValueError as e:
        logger.error(f"Critical sensors missing! {e}")
        return

    # 1. CALCUL DES LONGUEURS RÉELLES
    if mask is not None:
        lengths = np.sum(mask, axis=1).astype(int)
    else:
        logger.warning("Mask not found, inferring lengths from non-zero...")
        lengths = np.array([np.max(np.nonzero(sample[:, 0])) + 1 for sample in X])

    # 2. ALIGNEMENT TEMPOREL SUR L'ATTERRISSAGE (Right-Alignment virtuel)
    window_size = 100
    N, T, F = X.shape

    X_aligned_end = np.full((N, window_size, F), np.nan)
    final_values_vs = np.zeros(N)
    final_values_alt = np.zeros(N)

    for i in range(N):
        length = lengths[i]
        # Si le vol est vide ou buggé
        if length == 0: continue

        start_slice = max(0, length - window_size)
        segment = X[i, start_slice:length, :]

        actual_len = segment.shape[0]
        X_aligned_end[i, -actual_len:, :] = segment

        # Capture au point d'impact
        final_values_vs[i] = X[i, length - 1, idx_vs]
        final_values_alt[i] = X[i, length - 1, idx_alt]

    # --- PLOTTING ---
    plt.figure(figsize=(14, 10))

    # Subplot 1: Profil Radio Altitude
    plt.subplot(2, 2, 1)
    mean_profile = np.nanmean(X_aligned_end[:, :, idx_alt], axis=0)
    plt.plot(np.arange(-window_size, 0), mean_profile, color='red', linewidth=3, label='Mean Profile')

    # Trace 50 exemples
    count = 0
    for i in range(N):
        if count > 50: break
        if not np.isnan(X_aligned_end[i, -1, idx_alt]):
            plt.plot(np.arange(-window_size, 0), X_aligned_end[i, :, idx_alt], color='blue', alpha=0.1)
            count += 1

    plt.title("1. Alignment Check (Altitude @ End)")
    plt.xlabel("Time Steps relative to End")
    plt.ylabel("Feet")
    plt.grid(True)
    plt.legend()

    # Subplot 2: Correlation Glissante
    corrs_relative = []
    steps_relative = range(-window_size, 0)

    for t in range(window_size):
        vals_at_t = X_aligned_end[:, t, idx_vs]
        # On ne garde que les valeurs valides (pas NaN et pas les zéros de padding par sécurité si mask raté)
        valid_mask = ~np.isnan(vals_at_t)

        if np.sum(valid_mask) > 20:  # Au moins 20 vols pour une corrélation
            c = np.corrcoef(vals_at_t[valid_mask], y[valid_mask])[0, 1]
            corrs_relative.append(c)
        else:
            corrs_relative.append(0)

    plt.subplot(2, 2, 2)
    plt.plot(steps_relative, corrs_relative, color='green')
    plt.title("2. Physics Check: Correlation(IVV, VRTG)")
    plt.xlabel("Time Steps relative to End")
    plt.ylabel("Correlation")
    plt.grid(True)

    # Subplot 3: "Golden Feature" Check
    plt.subplot(2, 2, 3)
    plt.scatter(final_values_vs, y, alpha=0.3)
    plt.title(f"3. Final Sink Rate vs VRTG (Corr: {np.corrcoef(final_values_vs, y)[0, 1]:.2f})")
    plt.xlabel("Final Vertical Speed")
    plt.ylabel("Target VRTG")
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    # --- TEXT REPORT ---
    logger.info("-" * 30)
    avg_final_alt = np.mean(final_values_alt)
    logger.info(f"Avg Final Altitude: {avg_final_alt:.2f} ft")

    if avg_final_alt > 5.0:
        logger.warning(f">> ALERTE: L'altitude finale moyenne est de {avg_final_alt:.2f} ft.")
        logger.warning("   C'est la preuve que les données s'arrêtent AVANT l'impact.")
        logger.warning("   Action requise : Modifier '03_generate_datasets.py' pour inclure l'atterrissage.")
    else:
        logger.info(">> OK: L'altitude finale est proche de 0. Les données incluent l'impact.")


if __name__ == "__main__":
    main()