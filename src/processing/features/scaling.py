import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

from src.config import PADDING_VALUE


class MaskedStandardScaler(BaseEstimator, TransformerMixin):
    """
    Standard Scaler that handles padded sequences correctly.
    It computes Mean and Std ONLY on valid data (ignoring PADDING_VALUE).
    During transform, it ensures PADDING_VALUE is preserved strictly.
    """

    def __init__(self, padding_value: float = PADDING_VALUE):
        self.padding_value = padding_value
        self.mean_ = None
        self.scale_ = None
        self.var_ = None

    def fit(self, X: np.ndarray, y=None):
        """
        Compute the mean and std to be used for later scaling.
        X shape: (n_samples, n_timesteps, n_features) or (n_samples, n_features)
        """
        # Create a boolean mask of valid data
        # Use isclose to avoid floating point comparison issues
        mask = ~np.isclose(X, self.padding_value)

        # We need to compute stats per feature (last axis)
        n_features = X.shape[-1]

        self.mean_ = np.zeros(n_features)
        self.var_ = np.zeros(n_features)
        self.scale_ = np.zeros(n_features)

        for i in range(n_features):
            # Extract valid values for this feature across all samples and timesteps
            if X.ndim == 3:
                feature_slice = X[:, :, i]
                valid_values = feature_slice[mask[:, :, i]]
            else:
                feature_slice = X[:, i]
                valid_values = feature_slice[mask[:, i]]

            if valid_values.size > 0:
                self.mean_[i] = np.mean(valid_values)
                self.var_[i] = np.var(valid_values)
                self.scale_[i] = np.std(valid_values)
            else:
                # Fallback if a feature is entirely padding (should not happen)
                self.mean_[i] = 0.0
                self.scale_[i] = 1.0

        # Avoid division by zero for constant features
        self.scale_[self.scale_ == 0.0] = 1.0

        return self

    def transform(self, X: np.ndarray, y=None):
        """
        Perform standardization by centering and scaling.
        Preserves padding values exactly.
        """
        if self.mean_ is None:
            raise RuntimeError("Scaler is not fitted yet.")

        X_transformed = X.copy()

        # Identify padding locations
        mask_pad = np.isclose(X, self.padding_value)

        # Apply scaling: (X - mean) / std
        # We leverage broadcasting here
        if X.ndim == 3:
            X_transformed = (X_transformed - self.mean_) / self.scale_
        else:
            X_transformed = (X_transformed - self.mean_) / self.scale_

        # CRITICAL: Restore padding values that were mathematically garbled by the scaling
        X_transformed[mask_pad] = self.padding_value

        return X_transformed

    def fit_transform(self, X, y=None, **fit_params):
        return self.fit(X, y).transform(X)