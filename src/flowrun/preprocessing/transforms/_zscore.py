from dataclasses import dataclass

import numpy as np
from pydantic import field_validator
from scfit.registry import component

from flowrun.preprocessing.transforms._base import BaseTransform, TransformConfig, TransformParams

__all__ = ["ZScoreParams", "ZScoreTransform", "ZScoreTransformConfig"]


@dataclass
class ZScoreParams(TransformParams):
    """Parameters for Z-score normalization.

    These parameters are learned during the fit process and used for
    transformation and inverse transformation.

    Attributes
    ----------
        mean: Mean of each feature, shape (n_features,).
        std: Standard deviation of each feature, shape (n_features,).
        is_fitted: Whether the preprocessor has been fitted.
        n_features: Number of features in the fitted data.
    """

    mean: np.ndarray | None = None
    std: np.ndarray | None = None


class ZScoreTransform(BaseTransform):
    """Z-score normalization

    This preprocessor standardizes features by removing the mean and scaling
    to unit variance. The transformation is:
        X_transformed = (X - mean) / std

    Where mean and std are computed from the training data.

    Parameters
    ----------
        epsilon: Small value added to std to avoid division by zero.
            Default: 1e-8. Useful when a feature has zero variance.
    """

    _params_cls = ZScoreParams

    def __init__(self, epsilon: float = 1e-8) -> None:
        """Initialize ZScoreTransform.

        Args:
            epsilon (float): Small constant added to std to prevent division by zero.
        """
        super().__init__()
        if epsilon < 0:
            raise ValueError(f"epsilon must be non-negative, got {epsilon}.")
        self._epsilon = epsilon
        self._params: ZScoreParams | None = None

    def fit(self, X: np.ndarray) -> "ZScoreTransform":
        """Compute mean and std.

        Args:
            X: Input data array, shape (n_samples, n_features).
                Each row is a sample, each column is a feature.

        Returns
        -------
            Self
        """
        self._validate_input(X)

        mean = np.mean(X, axis=0)
        std = np.std(X, axis=0, ddof=0) + self._epsilon

        self._params = ZScoreParams(mean=mean, std=std, is_fitted=True, n_features=X.shape[1])

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform data to z-scores using fitted parameters.

        This applies the transformation: (X - mean) / std
        Each feature is centered (mean=0) and scaled (std=1).

        Args:
            X: Input data array, shape (n_samples, n_features).
                Must have same n_features as the data used in fit().

        Returns
        -------
            Z-Score array, shape (n_samples, n_features).
        """
        self.params._validate_fitted()
        self._validate_input(X)
        self.params._validate_n_features(X)

        return (X - self.params.mean) / self.params.std

    def inverse_transform(self, X_transformed: np.ndarray) -> np.ndarray:
        """Reverse z-score transformation to original scale.

        This applies the inverse transformation: X_transformed * std + mean

        Args:
            X_transformed: Standardized data array, shape (n_samples, n_features).

        Returns
        -------
            Reconstructed data array in original scale, shape (n_samples, n_features).
        """
        self.params._validate_fitted()
        self._validate_input(X_transformed)
        self.params._validate_n_features(X_transformed)

        return X_transformed * self.params.std + self.params.mean

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Convenience method that combines fit() and transform().

        Args:
            X: Input data array, shape (n_samples, n_features).

        Returns
        -------
            Standardized data array, shape (n_samples, n_features).
        """
        self.fit(X)
        return self.transform(X)


@component("transform.zscore")
class ZScoreTransformConfig(TransformConfig):
    """Portable config for :class:`ZScoreTransform`.

    Carries only the ``epsilon`` hyperparameter — the fitted ``mean`` and
    ``std`` arrays live on the runtime transform, never in the spec.

    ``epsilon`` is validated at construction time so a negative value (which
    would defeat the numerical-stability guard at :meth:`ZScoreTransform.fit`)
    fails before the training pipeline even starts.
    """

    epsilon: float = 1e-8

    @field_validator("epsilon")
    @classmethod
    def _epsilon_non_negative(cls, v: float) -> float:
        if v < 0:
            raise ValueError("epsilon must be non-negative")
        return v

    def build(self) -> ZScoreTransform:
        """Return a fresh, unfitted :class:`ZScoreTransform`."""
        return ZScoreTransform(epsilon=self.epsilon)
