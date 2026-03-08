import json
import sys
from pathlib import Path

import keras_tuner as kt
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

# Setup path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))

from src.config import DATASETS_DIR, PROCESSED_DIR, LOGS_DIR
from src.modeling.hypermodels import FlightHyperModel


def run_tuning(experiment_name: str, max_trials=20, executions_per_trial=1):
    print("========================================")
    print(f"   HYPERPARAMETER TUNING: {experiment_name}")
    print("========================================")

    # 1. Data Loading
    # Pointing to the global test dataset generated previously
    # Adapt path if necessary
    dataset_path = DATASETS_DIR / "test_run_global" / "dataset_dense.npz"
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    print(f"[INFO] Loading {dataset_path.name}...")
    data = np.load(dataset_path, allow_pickle=True)
    X_train, y_train = data['X_train'], data['y_train']
    X_val, y_val = data['X_val'], data['y_val']

    # 2. Normalization (Critical for convergence)
    # Re-creating scalers here for the experiment context
    print("[INFO] Normalizing features...")
    N, T, F = X_train.shape

    scaler_X = StandardScaler()
    # Fit on flattened Train set
    X_train_flat = X_train.reshape(-1, F)
    scaler_X.fit(X_train_flat)

    # Transform
    X_train = scaler_X.transform(X_train_flat).reshape(N, T, F)
    X_val = scaler_X.transform(X_val.reshape(-1, F)).reshape(X_val.shape)

    # Scale Target
    scaler_y = StandardScaler()
    y_train = scaler_y.fit_transform(y_train.reshape(-1, 1))
    y_val = scaler_y.transform(y_val.reshape(-1, 1))

    # 3. Tuner Configuration
    # BayesianOptimization is generally the most efficient for this topology
    tuner = kt.BayesianOptimization(
        hypermodel=FlightHyperModel(input_shape=(T, F)),
        objective="val_loss",
        max_trials=max_trials,  # Total number of combinations to test
        executions_per_trial=executions_per_trial,  # Average results to reduce variance
        directory=LOGS_DIR / "grid_search",
        project_name=experiment_name,
        overwrite=True  # Overwrites previous experiment with same name
    )

    tuner.search_space_summary()

    # 4. Starting Search
    print("\n[START] Starting Search Loop...")

    # Search-specific callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5),
        # TensorBoard to visualize curves for each trial
        tf.keras.callbacks.TensorBoard(log_dir=LOGS_DIR / "tensorboard" / f"{experiment_name}_tuning")
    ]

    tuner.search(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=30,  # Epochs per trial (enough to detect convergence)
        batch_size=32,
        callbacks=callbacks,
        verbose=1
    )

    # 5. Results
    print("\n[DONE] Tuning Complete.")
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

    print("\n--- BEST HYPERPARAMETERS ---")
    print(f"RNN Type: {best_hps.get('rnn_type')}")
    print(f"Num Layers: {best_hps.get('num_rnn_layers')}")
    print(f"Use CNN: {best_hps.get('use_cnn_block')}")
    print(f"Learning Rate: {best_hps.get('learning_rate'):.5f}")

    # Saving
    results_dir = PROCESSED_DIR / "models" / experiment_name
    results_dir.mkdir(parents=True, exist_ok=True)

    # Save config to JSON for reloading in training.py
    with open(results_dir / "best_hyperparameters.json", "w") as f:
        json.dump(best_hps.values, f, indent=4)

    print(f"[INFO] Best config saved to {results_dir / 'best_hyperparameters.json'}")


if __name__ == "__main__":
    # Launch an experiment
    run_tuning("Exp_Alpha2_Discovery", max_trials=10)