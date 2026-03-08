import json
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import randint

# Sklearn imports
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import RandomizedSearchCV, train_test_split

# TensorFlow imports
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, Callback
from tensorflow.keras.layers import LSTM, Dense, Dropout, Masking, Input, GlobalAveragePooling1D
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

# Project imports
from src.config import PADDING_VALUE
from src.processing.features.scaling import MaskedStandardScaler

# Configuration
logger = logging.getLogger(__name__)


# =============================================================================
# METRICS & UTILS
# =============================================================================
def r2_metric(y_true, y_pred):
    """
    Custom Keras metric for R2 Score (Coefficient of Determination).
    Note: Calculated batch-wise during training (approximation).
    """
    SS_res = K.sum(K.square(y_true - y_pred))
    SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
    return 1 - SS_res / (SS_tot + K.epsilon())


class FileLoggingCallback(Callback):
    """
    Logs Keras training metrics to the Python logger at the end of each epoch.
    """

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        # Format keys to be more readable (e.g. 'val_r2_metric' -> 'val_r2')
        metrics_str = " - ".join([f"{k}: {v:.4f}" for k, v in logs.items()])
        logger.info(f"Epoch {epoch + 1}: {metrics_str}")


# =============================================================================
# BASE CLASS
# =============================================================================
class BaseFeatureSelector(ABC):
    """
    Abstract Base Class for Feature Selection strategies involving Shadow Features.
    """

    def __init__(self, experiment_name: str, method_name: str):
        self.experiment_name = experiment_name
        self.method_name = method_name
        self.selected_features: List[str] = []
        self.rejected_features: List[str] = []
        self.feature_scores: Dict[str, float] = {}
        self.shadow_threshold: float = 0.0
        self.model_metrics: Dict[str, float] = {}
        self.validation_predictions: Optional[np.ndarray] = None

    def _add_shadow_feature(self, X: np.ndarray) -> np.ndarray:
        """
        Injects a Gaussian Noise feature (Shadow Feature) to determine the selection threshold.
        Handles padding correctly for 3D time-series data.
        """
        N = X.shape[0]

        if X.ndim == 2:
            # 2D Data: Simple concatenation
            noise = np.random.normal(0, 1, size=(N, 1))
            return np.hstack([X, noise])

        elif X.ndim == 3:
            # 3D Data: Concatenate along feature axis, respecting padding
            T = X.shape[1]
            noise = np.random.normal(0, 1, size=(N, T, 1))

            # Apply mask where X has padding
            mask = np.isclose(X[:, :, 0], PADDING_VALUE)
            noise[mask, 0] = PADDING_VALUE

            return np.concatenate([X, noise], axis=2)
        else:
            raise ValueError(f"Unsupported data dimensionality: {X.ndim}")

    def save_report(self, output_dir: Path) -> None:
        """Saves the selection results and metrics to a JSON file."""
        output_path = output_dir / f"selection_report_{self.method_name}.json"
        report = {
            "experiment": self.experiment_name,
            "method": self.method_name,
            "model_performance": self.model_metrics,
            "n_selected": len(self.selected_features),
            "shadow_threshold": self.shadow_threshold,
            "selected_features": self.selected_features,
            "rejected_features": self.rejected_features,
            "all_scores": self.feature_scores
        }
        try:
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=4)
            logger.info(f"Selection report saved to: {output_path}")
        except Exception as e:
            logger.error(f"Failed to save selection report: {e}")

    @abstractmethod
    def fit(self, X_train: np.ndarray, y_train: np.ndarray, feature_names: List[str],
            X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None, **kwargs) -> 'BaseFeatureSelector':
        pass


# =============================================================================
# RANDOM FOREST STRATEGY
# =============================================================================
class RandomForestSelector(BaseFeatureSelector):
    def __init__(self, experiment_name: str, optimize_params: bool = True, n_jobs: int = -1):
        super().__init__(experiment_name, "RandomForest_Shadow")
        self.optimize_params = optimize_params
        self.n_jobs = n_jobs
        self.scaler = MaskedStandardScaler(padding_value=PADDING_VALUE)
        self.model = None
        self.best_params = {
            "n_estimators": 200,
            "max_depth": 20,
            "min_samples_split": 2,
            "min_samples_leaf": 2,
            "max_features": "sqrt"
        }

    def _aggregate_time_series(self, X_3d: np.ndarray, feature_names: List[str]) -> Tuple[np.ndarray, List[str]]:
        """
        Flattens 3D time-series -> 2D features using statistical aggregation.
        """
        N, T, F = X_3d.shape
        X_flat_list = []
        new_names = []
        suffixes = ["_MEAN", "_STD", "_MIN", "_MAX", "_FINAL"]

        # Use masked array to ignore padding
        X_masked = np.ma.masked_values(X_3d, PADDING_VALUE)

        for i in range(F):
            sensor_data = X_masked[:, :, i]

            mean_val = np.mean(sensor_data, axis=1).filled(0.0)
            std_val = np.std(sensor_data, axis=1).filled(0.0)
            min_val = np.min(sensor_data, axis=1).filled(0.0)
            max_val = np.max(sensor_data, axis=1).filled(0.0)

            # Extract final valid value
            final_val = np.zeros(N)
            for j in range(N):
                valid_seq = sensor_data[j].compressed()
                if len(valid_seq) > 0:
                    final_val[j] = valid_seq[-1]
                else:
                    final_val[j] = 0.0

            X_flat_list.append(np.stack([mean_val, std_val, min_val, max_val, final_val], axis=1))
            new_names.extend([f"{feature_names[i]}{suf}" for suf in suffixes])

        return np.hstack(X_flat_list), new_names

    def _optimize_hyperparameters(self, X: np.ndarray, y: np.ndarray) -> Tuple[Dict, float]:
        logger.info("  > Optimizing Random Forest Hyperparameters...")
        param_dist = {
            'n_estimators': [100, 200, 300],
            'max_depth': [10, 20, 30, None],
            'min_samples_leaf': randint(1, 10),
            'min_samples_split': randint(2, 20),
            'max_features': ['sqrt', 'log2', None]
        }
        rf = RandomForestRegressor(random_state=42, n_jobs=self.n_jobs)
        search = RandomizedSearchCV(
            estimator=rf,
            param_distributions=param_dist,
            n_iter=15,
            cv=3,
            scoring='neg_mean_squared_error',
            n_jobs=self.n_jobs,
            random_state=42,
            verbose=0
        )
        search.fit(X, y)
        return search.best_params_, -search.best_score_

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, feature_names: List[str],
            X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None, **kwargs) -> 'RandomForestSelector':

        logger.info(f"Starting Random Forest Feature Selection (Optimize={self.optimize_params})...")

        # 1. Aggregation (3D -> 2D)
        if X_train.ndim == 3:
            X_agg, agg_names = self._aggregate_time_series(X_train, feature_names)
            if X_val is not None:
                X_val_agg, _ = self._aggregate_time_series(X_val, feature_names)
            else:
                X_val_agg = None
        else:
            X_agg, agg_names = X_train, feature_names
            X_val_agg = X_val

        # 2. Shadow Injection
        X_with_shadow = self._add_shadow_feature(X_agg)
        agg_names.append("RANDOM_SHADOW")

        if X_val_agg is not None:
            X_val_shadow = self._add_shadow_feature(X_val_agg)

        # 3. Scaling
        self.scaler = MaskedStandardScaler(padding_value=PADDING_VALUE)
        X_scaled = self.scaler.fit_transform(X_with_shadow)

        if X_val_agg is not None:
            X_val_scaled = self.scaler.transform(X_val_shadow)
        else:
            X_val_scaled = None

        # 4. Optimization
        cv_val_mse = None
        if self.optimize_params:
            self.best_params, cv_val_mse = self._optimize_hyperparameters(X_scaled, y_train)

        # 5. Training
        self.model = RandomForestRegressor(
            **self.best_params,
            oob_score=(X_val is None),
            random_state=42,
            n_jobs=self.n_jobs
        )

        logger.info(f"Fitting Final RF Model on {X_scaled.shape[1]} features...")
        self.model.fit(X_scaled, y_train)

        # 6. Metrics Calculation (MSE and R2)
        train_preds = self.model.predict(X_scaled)
        train_mse = mean_squared_error(y_train, train_preds)
        train_r2 = r2_score(y_train, train_preds)

        val_mse = "N/A"
        val_r2 = "N/A"

        if X_val_scaled is not None:
            # Explicit Validation
            val_preds = self.model.predict(X_val_scaled)
            self.validation_predictions = val_preds
            val_mse = mean_squared_error(y_val, val_preds)
            val_r2 = r2_score(y_val, val_preds)
            logger.info(f"  > Validation MSE: {val_mse:.4f} | R2: {val_r2:.4f}")

        elif cv_val_mse is not None:
            # CV approximation (R2 not available from RandomizedSearch default scoring unless specified)
            val_mse = cv_val_mse
            logger.info(f"  > Validation MSE (CV): {val_mse:.4f}")

        elif hasattr(self.model, 'oob_prediction_'):
            # OOB approximation
            val_mse = mean_squared_error(y_train, self.model.oob_prediction_)
            val_r2 = r2_score(y_train, self.model.oob_prediction_)
            logger.info(f"  > Validation MSE (OOB): {val_mse:.4f} | R2 (OOB): {val_r2:.4f}")

        self.model_metrics = {
            "train_mse": float(train_mse),
            "train_r2": float(train_r2),
            "val_mse": float(val_mse) if isinstance(val_mse, (float, int)) else val_mse,
            "val_r2": float(val_r2) if isinstance(val_r2, (float, int)) else val_r2
        }

        # 7. Importance Extraction
        importances = self.model.feature_importances_

        # Identify Shadow Threshold
        shadow_idx = agg_names.index("RANDOM_SHADOW")
        self.shadow_threshold = importances[shadow_idx]

        # Aggregate importance back to raw sensors
        raw_sensor_scores = {name: 0.0 for name in feature_names}
        for name, score in zip(agg_names, importances):
            if name == "RANDOM_SHADOW": continue
            for sensor in feature_names:
                if name.startswith(sensor):
                    raw_sensor_scores[sensor] += score
                    break

        self.feature_scores = dict(sorted(raw_sensor_scores.items(), key=lambda item: item[1], reverse=True))

        for sensor, score in self.feature_scores.items():
            if score > self.shadow_threshold:
                self.selected_features.append(sensor)
            else:
                self.rejected_features.append(sensor)

        return self


# =============================================================================
# LSTM STRATEGY (DYNAMIC)
# =============================================================================
class LSTMSelector(BaseFeatureSelector):
    def __init__(self, experiment_name: str, config: Dict[str, Any]):
        super().__init__(experiment_name, "LSTM_Permutation_Shadow")
        self.config = config
        self.model_config = config.get('model', {})
        self.selection_config = config.get('selection', {})

        self.epochs = self.selection_config.get('epochs', 50)
        self.batch_size = self.selection_config.get('batch_size', 32)

        self.scaler = MaskedStandardScaler(padding_value=PADDING_VALUE)
        self.history: Optional[Dict[str, List[float]]] = None
        self.model = None

    def _build_model_dynamic(self, input_shape: Tuple[int, int]) -> Sequential:
        """
        Constructs a Keras model dynamically based on configuration.
        """
        model = Sequential()

        # Input & Masking
        model.add(Input(shape=input_shape))
        model.add(Masking(mask_value=PADDING_VALUE))

        # Dynamic Layers
        layers_def = self.model_config.get('layers', [])
        if not layers_def:
            logger.warning("No layers defined in config! Using default architecture.")
            model.add(LSTM(32, return_sequences=False))
            model.add(Dense(1))

        for layer_cfg in layers_def:
            kwargs = layer_cfg.copy()
            l_type = kwargs.pop('type')

            if l_type == 'LSTM':
                model.add(LSTM(**kwargs))
            elif l_type == 'Dense':
                model.add(Dense(**kwargs))
            elif l_type == 'Dropout':
                if 'dropout' in kwargs: kwargs['rate'] = kwargs.pop('dropout')
                model.add(Dropout(**kwargs))
            elif l_type == 'GlobalAveragePooling1D':
                model.add(GlobalAveragePooling1D(**kwargs))
            else:
                logger.warning(f"Unsupported layer type: {l_type}")

        # Output Layer
        model.add(Dense(1, activation='linear'))

        # Compilation
        lr = self.model_config.get('learning_rate', 0.001)
        optimizer = Adam(learning_rate=lr)

        # Added r2_metric to the metrics list
        model.compile(
            optimizer=optimizer,
            loss=self.model_config.get('loss', 'mse'),
            metrics=['mae', 'mse', r2_metric]
        )
        return model

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, feature_names: List[str],
            X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None, **kwargs) -> 'LSTMSelector':

        logger.info(f"Starting LSTM Feature Selection (Padding Value: {PADDING_VALUE})...")

        if X_val is None or y_val is None:
            logger.warning("No validation set provided. Performing internal split.")
            X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
        else:
            logger.info(f"Using provided validation set: {X_val.shape[0]} samples.")

        # 2. Shadow Injection
        X_train_shadow = self._add_shadow_feature(X_train)
        X_val_shadow = self._add_shadow_feature(X_val)
        current_names = feature_names + ["RANDOM_SHADOW"]

        # 3. Scaling
        self.scaler.fit(X_train_shadow)
        X_train_scaled = self.scaler.transform(X_train_shadow)
        X_val_scaled = self.scaler.transform(X_val_shadow)

        # 4. Train Model
        N_train, T, F = X_train_scaled.shape
        logger.info(f"Training Dynamic LSTM on {N_train} samples with {F} features...")
        self.model = self._build_model_dynamic((T, F))

        callbacks = [
            EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=1),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1),
            FileLoggingCallback()
        ]

        history_obj = self.model.fit(
            X_train_scaled, y_train,
            validation_data=(X_val_scaled, y_val),
            epochs=self.epochs,
            batch_size=self.batch_size,
            callbacks=callbacks,
            verbose=1
        )
        self.history = history_obj.history

        # 5. Capture Metrics (Final evaluation)
        val_preds = self.model.predict(X_val_scaled, verbose=0)
        self.validation_predictions = val_preds.flatten()

        # Evaluate returns [loss, mae, mse, r2_metric] based on compile metrics
        final_scores = self.model.evaluate(X_val_scaled, y_val, verbose=0)
        # Assuming metric order: loss, mae, mse, r2_metric
        # Map them carefully by name if needed, but order is usually preserved

        # Safe extraction from history for R2 if evaluate order is ambiguous
        final_val_r2 = self.history.get('val_r2_metric', [-1])[-1]

        self.model_metrics = {
            "train_mse": float(history_obj.history['loss'][-1]),
            "val_mse": float(final_scores[0]),  # Loss is usually MSE
            "val_r2": float(final_val_r2)
        }

        # 6. Permutation Importance
        logger.info("Computing Permutation Importance on Validation Set...")
        baseline_score = final_scores[0]  # Use Loss (MSE) as baseline
        importances = {}

        for i, name in enumerate(current_names):
            X_val_corrupted = X_val_scaled.copy()
            np.random.shuffle(X_val_corrupted[:, :, i])
            # Evaluate only returns scalars
            corrupt_scores = self.model.evaluate(X_val_corrupted, y_val, verbose=0)
            score = corrupt_scores[0]  # MSE
            importances[name] = score - baseline_score

        # Determine Selection Threshold
        shadow_imp = importances.get("RANDOM_SHADOW", 0.0)
        self.shadow_threshold = max(shadow_imp, 0.0)

        sorted_feats = sorted(importances.items(), key=lambda x: x[1], reverse=True)
        self.feature_scores = dict(sorted_feats)

        self.selected_features = [
            name for name, score in sorted_feats
            if name != "RANDOM_SHADOW" and score > self.shadow_threshold
        ]
        self.rejected_features = [
            name for name, score in sorted_feats
            if name != "RANDOM_SHADOW" and score <= self.shadow_threshold
        ]

        return self

    def plot_learning_curves(self, output_dir: Path):
        """
        Plots Train vs Val history for Loss, MAE, and R2 Score.
        """
        if self.history is None:
            logger.warning("No history to plot.")
            return

        # Check available metrics in history
        has_r2 = 'r2_metric' in self.history

        # Setup grid
        cols = 3 if has_r2 else 2
        plt.figure(figsize=(6 * cols, 5))

        # 1. Loss (MSE)
        plt.subplot(1, cols, 1)
        plt.plot(self.history['loss'], label='Train Loss')
        plt.plot(self.history['val_loss'], label='Val Loss')
        plt.title('Model Loss (MSE)')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)

        # 2. MAE
        if 'mae' in self.history:
            plt.subplot(1, cols, 2)
            plt.plot(self.history['mae'], label='Train MAE')
            plt.plot(self.history['val_mae'], label='Val MAE')
            plt.title('Mean Absolute Error')
            plt.xlabel('Epochs')
            plt.legend()
            plt.grid(True)

        # 3. R2 Score (if available)
        if has_r2:
            plt.subplot(1, cols, 3)
            plt.plot(self.history['r2_metric'], label='Train R2')
            plt.plot(self.history['val_r2_metric'], label='Val R2')
            plt.title('R2 Score (Coeff. Determination)')
            plt.xlabel('Epochs')
            plt.ylabel('R2 Score')
            plt.legend(loc='lower right')
            plt.ylim(top=1.0)  # R2 max is 1.0
            plt.grid(True)

        output_path = output_dir / f"learning_curves_{self.method_name}.png"
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
        logger.info(f"Learning curves saved to: {output_path}")