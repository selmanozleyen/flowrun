from dataclasses import dataclass

import numpy as np
import scanpy as sc
from scfit.registry import component

from flowrun.preprocessing.transforms._base import BaseTransform, TransformConfig, TransformParams

__all__ = ["PCAParams", "PCATransform", "PCATransformConfig"]


@dataclass
class PCAParams(TransformParams):
    """Parameters for PCA transformation.

    These parameters are learned during the fit process through
    eigendecomposition of the covariance matrix.

    Attributes
    ----------
        mean: Mean of each feature, shape (n_features,).
            Data is centered before PCA.
        components: Principal component directions, shape (n_components, n_features).
            Each row is a principal component (eigenvector).
            Ordered by decreasing explained variance.
        explained_variance: Variance explained by each component, shape (n_components,).
            Eigenvalues of the covariance matrix.
        n_components: Number of principal components kept.
        is_fitted: Whether the preprocessor has been fitted.
        n_features: Number of features in the original data.
    """

    mean: np.ndarray | None = None
    components: np.ndarray | None = None
    explained_variance: np.ndarray | None = None
    n_components: int | None = None


class PCATransform(BaseTransform):
    """Principal Component Analysis for dimensionality reduction."""

    _params_cls = PCAParams

    def __init__(self, n_components: int | None = None) -> None:
        """Initialization

        Args:
            n_components: Number of principal components to keep.
                If None, keep all components (n_features).
                If int, keep that many components.
                Must be > 0 and <= n_features.
        """
        super().__init__()
        self.n_components = n_components
        self._params: PCAParams | None = None

    def fit(self, X: np.ndarray) -> "PCATransform":
        """Fit PCA by computing principal components.

        Args:
            X: Input data array, shape (n_samples, n_features)

        Returns
        -------
            Self
        """
        self._validate_input(X)
        X = X.astype(np.float64)

        _, n_features = X.shape
        if self.n_components is None:
            n_components = n_features
        else:
            if not isinstance(self.n_components, int):
                raise TypeError("n_components must be an integer or None")
            if self.n_components <= 0:
                raise ValueError("n_components must be positive")
            if self.n_components > n_features:
                raise ValueError(f"n_components ({self.n_components}) cannot be larger than n_features ({n_features})")
            n_components = self.n_components

        X_mean = np.array(np.mean(X, axis=0))
        X_centered = np.array(X - X_mean)
        X_pca, PCs, variance_ratio, variance = sc.pp.pca(
            X_centered, zero_center=False, return_info=True, n_comps=n_components
        )

        self._params = PCAParams(
            mean=X_mean,
            components=PCs,
            explained_variance=variance,
            n_components=n_components,
            is_fitted=True,
            n_features=X.shape[1],
        )

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform data to PC space.

        Args:
            X: Input data array, shape (n_samples, n_features).
                Must have same n_features as fitting data.

        Returns
        -------
            Transformed data in PC space, shape (n_samples, n_components).
        """
        self.params._validate_fitted()
        self._validate_input(X)
        self.params._validate_n_features(X)
        X = X.astype(np.float64)

        X_centered = np.array(X - self.params.mean)
        return X_centered @ self.params.components.T

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """Reconstruct data from PC space.

        Note: Lossy inverse transformation.

        Args:
            X: Data in PC space, shape (n_samples, n_components).

        Returns
        -------
            Reconstructed data, shape (n_samples, n_features).
        """
        self.params._validate_fitted()
        self._validate_input(X)

        if X.shape[1] != self.params.n_components:
            raise ValueError(
                f"Input has {X.shape[1]} components, but preprocessor "
                f"was fitted with {self.params.n_components} components."
            )

        return X @ self.params.components + self.params.mean

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Convenience method that combines fit() and transform().

        Args:
            X: Input data array, shape (n_samples, n_features).

        Returns
        -------
            Transformed data in PC space, shape (n_samples, n_components).
        """
        X = X.astype(np.float64)
        self.fit(X)
        return self.transform(X)

    def explained_variance_ratio(self) -> np.ndarray:
        """Get the proportion of variance explained by each component.

        This shows how much information (variance) each principal component
        captures. Useful for deciding how many components to keep.

        Returns
        -------
            Variance ratio for each component, shape (n_components,).
            Sum of all ratios <= 1.0

        Raises
        ------
            RuntimeError: If preprocessor has not been fitted.
        """
        self.params._validate_fitted()

        total_variance = np.sum(self.params.explained_variance)
        return self.params.explained_variance / total_variance


@component("transform.pca")
class PCATransformConfig(TransformConfig):
    """Portable config for :class:`PCATransform`.

    Carries only ``n_components`` — the fitted ``mean``, ``components`` and
    ``explained_variance`` arrays live on the runtime transform, never in the
    spec. When ``n_components is None``, the transform defaults to
    ``n_features`` at fit time.
    """

    n_components: int | None = None

    def build(self) -> PCATransform:
        """Return a fresh, unfitted :class:`PCATransform`."""
        return PCATransform(n_components=self.n_components)
