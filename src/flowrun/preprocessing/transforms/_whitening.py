"""Whitening transforms (ZCA / PCA / Cholesky).

Whitening rescales and reorients the feature axes so that the covariance of
the output is (approximately) the identity. Unlike PCA, whitening preserves
the feature dimension: the transform matrix ``W`` is ``(n_features, n_features)``.

Three strategies are supported, differing only in which ``W`` they produce:

* ``"pca"``      — ``W = diag(1/sqrt(λ + ε)) @ U.T``. Projects to PCA space
  and scales each component to unit variance. **Not** symmetric: the output
  basis is the PCA basis.
* ``"zca"``      — ``W = U @ diag(1/sqrt(λ + ε)) @ U.T``. The Mahalanobis /
  ZCA whitening. Symmetric: the output stays in the original coordinate frame,
  which is useful when downstream consumers care about feature identity.
* ``"cholesky"`` — ``W = inv(L)`` with ``L = cholesky(Σ)``. Triangular
  whitening; the fastest of the three but the resulting basis is arbitrary
  (depends on the Cholesky pivoting) and usually only useful as a
  pre-conditioner.

``ε`` is added to the singular values (PCA / ZCA) or as a relative jitter on
the covariance diagonal (Cholesky) so the matrices remain well-conditioned
even when the data is rank-deficient.

See :class:`WhiteningTransformConfig` for the portable, serializable side of
the same transformation, and :class:`BaseTransform` for the AnnData adapter
(``fit_adata`` / ``transform_adata`` / ``fit_transform_adata``).
"""

from dataclasses import dataclass
from typing import Literal

import numpy as np
from pydantic import field_validator
from scfit.registry import component
from scipy.linalg import cholesky

from flowrun.preprocessing.transforms._base import (
    BaseTransform,
    TransformConfig,
    TransformParams,
)

__all__ = [
    "WhiteningParams",
    "WhiteningTransform",
    "WhiteningTransformConfig",
    "compute_whitening_matrix",
]

WhiteningMethod = Literal["zca", "pca", "cholesky"]
_VALID_METHODS: tuple[str, ...] = ("zca", "pca", "cholesky")


# --------------------------------------------------------------------------- #
# Params
# --------------------------------------------------------------------------- #
@dataclass
class WhiteningParams(TransformParams):
    """Fitted parameters for a whitening transform.

    Attributes
    ----------
        mean: Feature-wise mean of the fitted data, shape ``(n_features,)``.
            The data is centered before whitening.
        W: Forward whitening matrix, shape ``(n_features, n_features)``.
            ``X_whitened = (X - mean) @ W.T``.
        iW: Inverse whitening matrix, shape ``(n_features, n_features)``.
            ``X = X_whitened @ iW.T + mean``.
        is_fitted: Whether the transform was fitted.
        n_features: Number of features in the fitted data.
    """

    mean: np.ndarray | None = None
    W: np.ndarray | None = None
    iW: np.ndarray | None = None


# --------------------------------------------------------------------------- #
# Runtime transform
# --------------------------------------------------------------------------- #
class WhiteningTransform(BaseTransform):
    """Whitening transform (ZCA / PCA / Cholesky).

    Parameters
    ----------
        method: One of ``"zca"``, ``"pca"`` or ``"cholesky"``.
        epsilon: Regularization added to the singular values (PCA / ZCA) or as
            a relative jitter on the covariance diagonal (Cholesky). Keeps the
            whitening matrices well-conditioned when ``Σ`` is (nearly) singular.
    """

    _params_cls = WhiteningParams

    def __init__(
        self,
        method: WhiteningMethod = "zca",
        epsilon: float = 1e-5,
    ) -> None:
        super().__init__()
        if method not in _VALID_METHODS:
            raise ValueError(f"Unknown method: {method!r}. Use one of {_VALID_METHODS}.")
        if epsilon < 0:
            raise ValueError(f"epsilon must be non-negative, got {epsilon}.")
        self._method: WhiteningMethod = method
        self._epsilon: float = float(epsilon)
        self._params: WhiteningParams | None = None

    @property
    def method(self) -> WhiteningMethod:
        """Whitening strategy used by this instance."""
        return self._method

    @property
    def epsilon(self) -> float:
        """Regularization strength."""
        return self._epsilon

    def fit(self, X: np.ndarray) -> "WhiteningTransform":
        """Fit the whitening matrix on the data.

        Args:
            X: Input data array, shape ``(n_samples, n_features)``.

        Returns
        -------
            Self
        """
        self._validate_input(X)
        X = X.astype(np.float64)

        mean = np.array(np.mean(X, axis=0))
        X_centered = X - mean
        # ``np.cov`` returns a 0-d array for a single feature; promote so the
        # linear algebra below stays 2-D regardless of ``n_features``.
        cov = np.atleast_2d(np.cov(X_centered, rowvar=False))

        if self._method in ("pca", "zca"):
            # ``cov`` is symmetric PSD, so U == V and its sign is irrelevant
            # for the whitening matrix (a sign flip in U cancels in W W^T).
            U, S, _ = np.linalg.svd(cov)
            sqrt_S = np.sqrt(S + self._epsilon)
            inv_sqrt_S = 1.0 / sqrt_S

            if self._method == "pca":
                # W = diag(1/sqrt(λ+ε)) @ U.T,  iW = U @ diag(sqrt(λ+ε))
                W = (U * inv_sqrt_S).T
                iW = U * sqrt_S
            else:  # zca
                # W = U @ diag(1/sqrt(λ+ε)) @ U.T,  iW = U @ diag(sqrt(λ+ε)) @ U.T
                W = (U * inv_sqrt_S) @ U.T
                iW = (U * sqrt_S) @ U.T
        else:  # cholesky
            n = cov.shape[0]
            # Scale-relative jitter so ``ε`` means the same thing whether the
            # data is on the unit scale or on counts; ``ε = 0`` reproduces the
            # classic un-jittered Cholesky decomposition.
            trace = float(np.trace(cov))
            scale = trace / n if trace > 0 else 1.0
            cov_reg = cov + self._epsilon * scale * np.eye(n)
            L = cholesky(cov_reg, lower=True)
            W = np.linalg.inv(L)
            iW = L

        self._params = WhiteningParams(
            mean=mean,
            W=W,
            iW=iW,
            is_fitted=True,
            n_features=X.shape[1],
        )
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Whiten the data using the fitted parameters.

        Args:
            X: Input data array, shape ``(n_samples, n_features)``.

        Returns
        -------
            Whitened data array, shape ``(n_samples, n_features)``.
        """
        self.params._validate_fitted()
        self._validate_input(X)
        self.params._validate_n_features(X)
        X = X.astype(np.float64)

        return (X - self.params.mean) @ self.params.W.T

    def inverse_transform(self, X_whitened: np.ndarray) -> np.ndarray:
        """Reverse the whitening transform.

        Note: exact only up to the regularizer ``ε``. With ``ε > 0`` this is
        a lossy inverse in the same sense as PCA — the ``ε``-jittered
        directions lose their exact scaling, so ``X_reconstructed ≈ X`` when
        the data has zero-variance features.

        Args:
            X_whitened: Whitened data, shape ``(n_samples, n_features)``.

        Returns
        -------
            Data in the original feature space, shape ``(n_samples, n_features)``.
        """
        self.params._validate_fitted()
        self._validate_input(X_whitened)
        self.params._validate_n_features(X_whitened)
        X_whitened = X_whitened.astype(np.float64)

        return X_whitened @ self.params.iW.T + self.params.mean

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Convenience method that combines :meth:`fit` and :meth:`transform`."""
        self.fit(X)
        return self.transform(X)


# --------------------------------------------------------------------------- #
# Portable config
# --------------------------------------------------------------------------- #
@component("transform.whitening")
class WhiteningTransformConfig(TransformConfig):
    """Portable config for :class:`WhiteningTransform`.

    Carries only the ``method`` selector and the ``epsilon`` regularizer — the
    fitted ``mean``, ``W`` and ``iW`` arrays live on the runtime transform,
    never in the spec. Both fields round-trip through ``to_spec`` /
    ``from_spec`` as plain scalars, so the config is safe to embed in a
    ``ModelConfig`` written by ``Model.save_pretrained``.

    ``epsilon`` is validated at construction time so a negative value (which
    would make the whitening matrix non-PSD) fails at config time rather than
    at fit time.
    """

    method: WhiteningMethod = "zca"
    epsilon: float = 1e-5

    @field_validator("epsilon")
    @classmethod
    def _epsilon_non_negative(cls, v: float) -> float:
        if v < 0:
            raise ValueError("epsilon must be non-negative")
        return v

    def build(self) -> WhiteningTransform:
        """Return a fresh, unfitted :class:`WhiteningTransform`."""
        return WhiteningTransform(method=self.method, epsilon=self.epsilon)


# --------------------------------------------------------------------------- #
# Functional shortcut (mirrors the standalone helper)
# --------------------------------------------------------------------------- #
def compute_whitening_matrix(
    X: np.ndarray,
    epsilon: float = 1e-5,
    method: WhiteningMethod = "zca",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(W, iW, mean)`` for the requested whitening strategy.

    Thin functional wrapper around :class:`WhiteningTransform` for callers
    that only want the matrices and not the runtime object::

        W, iW, mean = compute_whitening_matrix(X, epsilon=1e-5, method="zca")
        X_whitened = (X - mean) @ W.T
        X_back = X_whitened @ iW.T + mean

    See :class:`WhiteningTransform` for the definition of each matrix per
    method.
    """
    transform = WhiteningTransform(method=method, epsilon=epsilon).fit(X)
    return transform.params.W, transform.params.iW, transform.params.mean
