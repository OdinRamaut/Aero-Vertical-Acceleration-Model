import sys
import numpy as np
import logging
from pathlib import Path
from sklearn.model_selection import KFold

# =============================================================================
# 1. SETUP & PATHS
# =============================================================================
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.config import DATASETS_DIR, LOGS_DIR
from src.processing.features.selection import RandomForestSelector, LSTMSelector
from src.processing.features.stability import StabilityFeatureSelector

# =============================================================================
# 2. CONFIGURATION
# =============================================================================
EXPERIMENT_NAME = "Exp_06_Landing_VRTG_Padding_updated"
DATASET_PATH = DATASETS_DIR / EXPERIMENT_NAME / "dataset_dense.npz"

# Selection Parameters
SELECTION_MODE = "LSTM"  # Options: "RF" or "LSTM"
STABILITY_THRESHOLD = 0.6
N_FOLDS = 5

# LSTM Specific Configuration
LSTM_CONFIG = {
    'selection': {
        'epochs': 60,  # Augmenter un peu pour laisser le temps de converger
        'batch_size': 32
    },
    'model': {
        'learning_rate': 0.0005,  # Réduire légèrement le LR (plus stable)
        'loss': 'mse',
        'layers': [
            # Couche 1 : Plus large et garde la séquence temporelle
            {'type': 'LSTM', 'units': 64, 'return_sequences': True},

            # Ajout de Dropout pour éviter l'overfitting sur les nouvelles capacités
            {'type': 'Dropout', 'dropout': 0.2},

            # Couche 2 : Comprime intelligemment la séquence
            # Note : Si vous n'avez pas codé GlobalAveragePooling1D, mettez un 2ème LSTM
            {'type': 'LSTM', 'units': 32, 'return_sequences': False},

            # Couche Dense finale
            {'type': 'Dense', 'units': 1}
        ]
    }
}

# =============================================================================
# 3. LOGGING SETUP
# =============================================================================
exp_log_dir = LOGS_DIR / EXPERIMENT_NAME
exp_log_dir.mkdir(parents=True, exist_ok=True)
log_file_path = exp_log_dir / f"stability_selection_{SELECTION_MODE}.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(log_file_path, mode='w')
    ],
    force=True
)
logger = logging.getLogger(__name__)


def load_data(path: Path):
    """
    Loads the dataset. Merges Train and Val for Global Cross-Validation.
    """
    if not path.exists():
        logger.critical(f"Dataset not found at {path}")
        sys.exit(1)

    logger.info(f"Loading dataset from: {path.name}")
    data = np.load(path, allow_pickle=True)

    # Merge Train and Val for Cross-Validation (Stability Selection uses its own splits)
    if 'X_val' in data and data['X_val'].shape[0] > 0:
        logger.info("Merging existing Train and Val sets for Global Cross-Validation...")
        X = np.concatenate([data['X_train'], data['X_val']], axis=0)
        y = np.concatenate([data['y_train'], data['y_val']], axis=0)
    else:
        X = data['X_train']
        y = data['y_train']

    raw_names = data['feature_names']
    feature_names = [f.decode('utf-8') if isinstance(f, bytes) else str(f) for f in raw_names]

    logger.info(f"Total Data for Stability CV: {X.shape} samples, {len(feature_names)} features")
    return X, y, feature_names


def main():
    logger.info("==================================================")
    logger.info(f"   STABILITY SELECTION PIPELINE (Mode: {SELECTION_MODE})")
    logger.info("==================================================")

    # 1. Load Data
    X, y, names = load_data(DATASET_PATH)

    # 2. Configure Strategy
    output_dir = DATASETS_DIR / EXPERIMENT_NAME

    # K-Fold Strategy (Shuffle=True is critical if samples are independent flights)
    cv_strategy = KFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

    if SELECTION_MODE == "RF":
        selector_cls = RandomForestSelector
        selector_params = {
            "optimize_params": True,  # Optimization runs INSIDE each fold
            "n_jobs": -1
        }

    elif SELECTION_MODE == "LSTM":
        selector_cls = LSTMSelector
        selector_params = {
            "config": LSTM_CONFIG
        }


    else:
        logger.error(f"Unknown mode: {SELECTION_MODE}")
        sys.exit(1)

    # 3. Initialize Stability Orchestrator
    stability_selector = StabilityFeatureSelector(
        selector_cls=selector_cls,
        selector_params=selector_params,
        cv=cv_strategy,
        experiment_name=EXPERIMENT_NAME,
        threshold=STABILITY_THRESHOLD,
    )

    # 4. Execute
    try:
        stability_selector.fit(X, y, feature_names=names)
        stability_selector.save_report(output_dir)

        # 5. Console Summary
        scores = stability_selector.stability_scores_
        g_metrics = stability_selector.global_metrics_

        print("\n" + "-" * 60)
        print(f"STABILITY SELECTION COMPLETE ({SELECTION_MODE})")
        print("-" * 60)

        # --- NEW: Print Global Performance Metrics ---
        if g_metrics:
            print(f"Global Model Performance (Out-Of-Fold):")
            print(f"  > R² Score : {g_metrics.get('global_oof_r2', -99):.4f}")
            print(f"  > RMSE     : {g_metrics.get('global_oof_rmse', -1):.4f}")
            print("-" * 60)
        # ---------------------------------------------

        print(f"Selected Features (Threshold >= {STABILITY_THRESHOLD}):")
        print(f"{'Feature Name':<30} | {'Score':<10} | {'Status'}")
        print("-" * 60)

        for i, (feat, score) in enumerate(scores.items()):
            if i >= 15: break  # Show top 15 only
            status = "SELECTED" if score >= STABILITY_THRESHOLD else "REJECTED"
            # Color code slightly if supported, otherwise plain text
            print(f"{feat:<30} | {score:.2f}       | {status}")

        print("-" * 60)
        print(f"Full report saved to: {output_dir}")
        print(f"Logs saved to  : {log_file_path}")

    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()