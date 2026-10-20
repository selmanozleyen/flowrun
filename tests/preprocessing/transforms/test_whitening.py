"""Tests for :mod:`flowrun.preprocessing.transforms._whitening`."""

import numpy as np
import pytest
from anndata import AnnData
from pydantic import ValidationError

from flowrun.preprocessing.transforms import (
    WhiteningTransform,
    WhiteningTransformConfig,
)

METHODS: tuple[str, ...] = ("zca", "pca", "cholesky")


def _data(n_samples: int = 256, n_features: int = 6, seed: int = 0) -> np.ndarray:
    """Data with a controlled, well-conditioned covariance spectrum λ_i = 1..n_features.

    Drawing ``A`` from IID normal (as an earlier version did) leaves the
    smallest sample eigenvalue unbounded below; with ``n_features = 6`` it can
    land near zero and inflate the ``ε`` residual past the ``1e-4`` tolerance
    the whitening property test uses. An explicit spectrum pins
    ``λ_min = 1``, so the residual is bounded by ``ε / 1 = 1e-5`` (PCA / ZCA)
    or ``ε · trace(Σ)/n / 1 ≈ 3.5e-5`` (Cholesky) — comfortably inside
    tolerance.
    """
    rng = np.random.default_rng(seed)
    Q, _ = np.linalg.qr(rng.normal(size=(n_features, n_features)))
    spectrum = np.arange(1, n_features + 1, dtype=float)  # 1, 2, ..., n_features
    L = Q @ np.diag(np.sqrt(spectrum))
    return rng.normal(size=(n_samples, n_features)) @ L.T


def _adata(X: np.ndarray) -> AnnData:
    return AnnData(X.copy())


# --------------------------- whitening property ----------------------------


@pytest.mark.parametrize("method", METHODS)
def test_output_has_identity_covariance(method: str) -> None:
    """Whitened data should have unit covariance (up to ``ε``)."""
    X = _data()
    Y = WhiteningTransform(method=method).fit_transform(X)

    Y_centered = Y - Y.mean(axis=0)
    cov = np.atleast_2d(np.cov(Y_centered, rowvar=False))
    np.testing.assert_allclose(cov, np.eye(X.shape[1]), atol=1e-4)


@pytest.mark.parametrize("method", METHODS)
def test_inverse_recovers_original(method: str) -> None:
    X = _data()
    xf = WhiteningTransform(method=method).fit(X)
    X_back = xf.inverse_transform(xf.transform(X))
    np.testing.assert_allclose(X_back, X, atol=1e-6)


@pytest.mark.parametrize("method", METHODS)
def test_epsilon_zero_yields_exact_whitening(method: str) -> None:
    """With ``epsilon=0`` and invertible Σ, ``W Σ Wᵀ = I`` holds exactly.

    The whitening property is exact for all three methods and independent of
    the data spectrum, which makes it a sharper regression test than
    ``W Wᵀ = I`` (that identity only holds for a truly identity cov — never
    what a finite random sample produces).
    """
    X = _data()
    xf = WhiteningTransform(method=method, epsilon=0.0).fit(X)

    cov = np.atleast_2d(np.cov(X - X.mean(axis=0), rowvar=False))
    W = xf.params.W
    np.testing.assert_allclose(W @ cov @ W.T, np.eye(X.shape[1]), atol=1e-10)


# --------------------------- method-specific properties -------------------


def test_zca_matrix_is_symmetric() -> None:
    X = _data()
    xf = WhiteningTransform(method="zca").fit(X)
    np.testing.assert_allclose(xf.params.W, xf.params.W.T, atol=1e-10)
    np.testing.assert_allclose(xf.params.iW, xf.params.iW.T, atol=1e-10)


def test_pca_matrix_decorrelates_columns() -> None:
    X = _data()
    xf = WhiteningTransform(method="pca").fit(X)
    Y = xf.transform(X)
    np.testing.assert_allclose(np.cov(Y, rowvar=False), np.eye(X.shape[1]), atol=1e-4)


def test_cholesky_inverse_is_lower_triangular() -> None:
    X = _data()
    xf = WhiteningTransform(method="cholesky").fit(X)
    # ``iW = L`` (lower triangular) — verify upper triangle is zero.
    np.testing.assert_allclose(np.triu(xf.params.iW, k=1), 0.0, atol=1e-10)


# --------------------------- AnnData adapter ------------------------------


def test_fit_and_transform_adata_writes_to_out_obsm() -> None:
    X = _data(n_samples=64)
    adata = _adata(X)

    xf = WhiteningTransformConfig(method="zca").build()
    xf.fit_adata(adata, layer=None)  # fit on X
    xf.transform_adata(adata, out_obsm="X_whitened", inplace=True)

    assert "X_whitened" in adata.obsm
    assert adata.obsm["X_whitened"].shape == X.shape
    # Input X is untouched.
    np.testing.assert_array_equal(adata.X, X)


def test_build_and_fit_adata_one_shot() -> None:
    X = _data(n_samples=64)
    adata = _adata(X)
    xf = WhiteningTransformConfig(method="pca").build_and_fit(adata)
    out = xf.transform_adata(adata, out_obsm="X_pca_whitened", inplace=False)

    assert out.obsm["X_pca_whitened"].shape == X.shape
    assert "X_pca_whitened" not in adata.obsm  # copy, original untouched


# --------------------------- config round-trip ----------------------------


@pytest.mark.parametrize("method", METHODS)
def test_config_round_trips(method: str) -> None:
    cfg = WhiteningTransformConfig(method=method, epsilon=1e-7)
    rebuilt = WhiteningTransformConfig.from_spec(cfg.to_spec())
    assert rebuilt == cfg


def test_config_rejects_unknown_method() -> None:
    with pytest.raises(ValidationError):
        WhiteningTransformConfig(method="bogus")  # type: ignore[arg-type]


def test_config_rejects_negative_epsilon() -> None:
    """A negative epsilon would make the whitening matrix non-PSD."""
    with pytest.raises(ValidationError):
        WhiteningTransformConfig(epsilon=-1.0)


def test_transform_rejects_unknown_method() -> None:
    with pytest.raises(ValueError, match="Unknown method"):
        WhiteningTransform(method="bogus")  # type: ignore[arg-type]
