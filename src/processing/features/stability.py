import json
import logging
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Dict, Type, Any, Optional
from sklearn.model_selection import BaseCrossValidator
from sklearn.metrics import mean_squared_error, r2_score

# Import base class
from src.processing.features.selection import BaseFeatureSelector

logger = logging.getLogger(__name__)


class StabilityFeatureSelector:
    """
    Orchestrates a Stability Selection process using Cross-Validation (Out-Of-Fold Analysis).

    This class wraps a base feature selector, executes it across folds, and aggregates:
    1. Feature Stability (Frequency of selection)
    2. Model Performance (MSE, R2 via OOF predictions)

    Attributes:
        stability_scores_ (Dict[str, float]): Stability score (0.0 to 1.0) for each feature.
        selected_features_ (List[str]): Features that meet the stability threshold.
        detailed_report_ (Dict): Complete logs including per-fold metrics.
        global_metrics_ (Dict): Aggregated performance metrics (Mean MSE, Global R2).
    """

    def __init__(self,
                 selector_cls: Type[BaseFeatureSelector],
                 selector_params: Dict[str, Any],
                 cv: BaseCrossValidator,
                 experiment_name: str,
                 threshold: float = 0.6,
                 ):
        """
        Args:
            selector_cls: The class of the selector (e.g., RandomForestSelector).
            selector_params: Config dict for the selector.
            cv: Scikit-learn cross-validator.
            experiment_name: Label for logging.
            threshold: Min stability score (0.6 = 60%) to select a feature.
        """
        self.selector_cls = selector_cls
        self.selector_params = selector_params
        self.cv = cv
        self.experiment_name = experiment_name
        self.threshold = threshold

        # Results container
        self.stability_scores_: Dict[str, float] = {}
        self.selected_features_: List[str] = []
        self.rejected_features_: List[str] = []
        self.detailed_report_: Dict[str, Any] = {}
        self.global_metrics_: Dict[str, float] = {}

        # OOF Data Containers
        self.oof_y_true_: List[float] = []
        self.oof_y_pred_: List[float] = []

    def fit(self, X: np.ndarray, y: np.ndarray, feature_names: List[str]) -> 'StabilityFeatureSelector':
        """Executes the CV pipeline, aggregates votes and predictions."""
        logger.info(f"Starting Stability Selection (CV={self.cv.get_n_splits()}) on {len(feature_names)} features.")

        # 1. Generate Splits
        splits = list(self.cv.split(X, y))
        n_splits = len(splits)

        # 2. Sequential Execution
        results = []
        for k, (train_idx, val_idx) in enumerate(splits):
            logger.info(f"Processing Fold {k + 1}/{n_splits}...")

            # Appel direct synchrone
            fold_result = self._process_fold(
                k + 1, train_idx, val_idx, X, y, feature_names
            )
            results.append(fold_result)

        # 3. Aggregation
        feature_counts = {name: 0 for name in feature_names}
        fold_details = []

        # Reset OOF containers
        self.oof_y_true_ = []
        self.oof_y_pred_ = []

        mse_scores = []

        for res in results:
            if res["status"] == "FAILED":
                logger.error(f"Skipping failed fold {res['fold']}")
                continue

            # A. Feature Votes
            for feat in res['selected']:
                if feat in feature_counts:
                    feature_counts[feat] += 1

            # B. Performance Aggregation
            if res["val_mse"] is not None:
                mse_scores.append(res["val_mse"])

            # C. OOF Predictions Collection
            # We extend the global lists with this fold's validation data
            if res["y_true"] is not None and res["y_pred"] is not None:
                self.oof_y_true_.extend(res["y_true"])
                self.oof_y_pred_.extend(res["y_pred"])

            # Clean heavy arrays from report to save disk space
            res_clean = res.copy()
            del res_clean["y_true"]
            del res_clean["y_pred"]
            fold_details.append(res_clean)

        # 4. Compute Stability Scores
        self.stability_scores_ = {
            name: count / n_splits
            for name, count in feature_counts.items()
        }

        # Filter & Sort
        self.selected_features_ = [
            name for name, score in self.stability_scores_.items()
            if score >= self.threshold
        ]

        self.rejected_features_ = [
            name for name, score in self.stability_scores_.items()
            if score < self.threshold
        ]


        self.stability_scores_ = dict(sorted(
            self.stability_scores_.items(), key=lambda item: item[1], reverse=True
        ))

        # 5. Compute Global Performance Metrics
        global_mse = np.mean(mse_scores) if mse_scores else -1.0
        global_std_mse = np.std(mse_scores) if mse_scores else 0.0

        # Global R2 (calculated on concatenated OOF predictions)
        if len(self.oof_y_true_) > 0:
            global_r2 = r2_score(self.oof_y_true_, self.oof_y_pred_)
            global_rmse = np.sqrt(mean_squared_error(self.oof_y_true_, self.oof_y_pred_))
        else:
            global_r2 = -999.0
            global_rmse = -1.0

        self.global_metrics_ = {
            "mean_mse_folds": float(global_mse),
            "std_mse_folds": float(global_std_mse),
            "global_oof_rmse": float(global_rmse),
            "global_oof_r2": float(global_r2)
        }

        # 6. Build Report
        self.detailed_report_ = {
            "experiment": self.experiment_name,
            "method": f"Stability_{self.selector_cls.__name__}",
            "n_folds": n_splits,
            "n_samples": X.shape[0],
            "threshold": self.threshold,
            "global_metrics": self.global_metrics_,
            "summary": {
                "n_selected": len(self.selected_features_),
                "n_rejected": len(self.rejected_features_),
                "selected_features": self.selected_features_,
                "rejected_features": self.rejected_features_,
                "stability_scores": self.stability_scores_
            },
            "folds": fold_details
        }

        logger.info(f"Stability Complete. Selected: {len(self.selected_features_)}. Global R2: {global_r2:.4f}")
        return self

    def _process_fold(self, fold_id: int, train_idx: np.ndarray, val_idx: np.ndarray,
                      X: np.ndarray, y: np.ndarray, feature_names: List[str]) -> Dict[str, Any]:
        """Worker method for a single fold."""
        fold_exp_name = f"{self.experiment_name}_Fold_{fold_id}"

        # Instantiate selector
        selector = self.selector_cls(experiment_name=fold_exp_name, **self.selector_params)

        X_train_fold, y_train_fold = X[train_idx], y[train_idx]
        X_val_fold, y_val_fold = X[val_idx], y[val_idx]

        try:
            selector.fit(
                X_train=X_train_fold, y_train=y_train_fold,
                feature_names=feature_names,
                X_val=X_val_fold, y_val=y_val_fold
            )

            # Retrieve Predictions safely
            y_pred_fold = selector.validation_predictions
            if y_pred_fold is None:
                # Fallback if selector didn't store preds
                logger.warning(f"Fold {fold_id}: No validation predictions found.")
                y_pred_fold = np.zeros_like(y_val_fold)

            return {
                "fold": fold_id,
                "status": "SUCCESS",
                "n_selected": len(selector.selected_features),
                "shadow_threshold": getattr(selector, 'shadow_threshold', None),
                "val_mse": selector.model_metrics.get('val_mse', None),
                "selected": selector.selected_features,
                "rejected": selector.rejected_features,
                "y_true": y_val_fold.tolist(),  # Convert to list for serialization/transfer
                "y_pred": y_pred_fold.tolist()
            }

        except Exception as e:
            logger.error(f"Fold {fold_id} failed: {e}")
            return {
                "fold": fold_id, "status": "FAILED", "error": str(e),
                "selected": [], "y_true": None, "y_pred": None
            }

    def save_report(self, output_dir: Path) -> None:
        """Saves JSON report and Generates Performance Plots."""
        if not self.detailed_report_:
            logger.warning("No report to save. Call fit() first.")
            return

        # 1. Save JSON
        filename = f"stability_report_{self.selector_cls.__name__}.json"
        output_path = output_dir / filename
        try:
            with open(output_path, 'w') as f:
                json.dump(self.detailed_report_, f, indent=4)
            logger.info(f"Stability report saved to: {output_path}")
        except Exception as e:
            logger.error(f"Failed to save report: {e}")

        # 2. Generate Plots
        self._plot_performance(output_dir)

    def _plot_performance(self, output_dir: Path):
        """Generates a Predicted vs Actual plot + Residuals."""
        if not self.oof_y_true_:
            return

        y_true = np.array(self.oof_y_true_)
        y_pred = np.array(self.oof_y_pred_)

        # Metrics
        r2 = self.global_metrics_.get('global_oof_r2', 0)
        rmse = self.global_metrics_.get('global_oof_rmse', 0)

        fig, ax = plt.subplots(1, 2, figsize=(16, 6))

        # Subplot 1: Scatter (Predicted vs True)
        ax[0].scatter(y_true, y_pred, alpha=0.5, s=10, c='blue', label='Data Points')

        # Identity Line (Perfect prediction)
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        ax[0].plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Identity (x=y)')

        ax[0].set_title(f'OOF Prediction Accuracy\n$R^2$={r2:.3f}, RMSE={rmse:.3f}')
        ax[0].set_xlabel('True Values')
        ax[0].set_ylabel('Predicted Values')
        ax[0].legend()
        ax[0].grid(True, alpha=0.3)

        # Subplot 2: Residual Distribution
        residuals = y_true - y_pred
        ax[1].hist(residuals, bins=50, color='purple', alpha=0.7, edgecolor='black')
        ax[1].axvline(0, color='r', linestyle='--', linewidth=2)
        ax[1].set_title(f'Residuals Distribution\nMean={np.mean(residuals):.3f}, Std={np.std(residuals):.3f}')
        ax[1].set_xlabel('Error (True - Pred)')
        ax[1].set_ylabel('Frequency')
        ax[1].grid(True, alpha=0.3)

        plot_path = output_dir / f"stability_performance_{self.selector_cls.__name__}.png"
        plt.tight_layout()
        plt.savefig(plot_path)
        plt.close()
        logger.info(f"Performance plot saved to: {plot_path}")