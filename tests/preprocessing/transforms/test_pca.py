"""Tests for :mod:`flowrun.preprocessing.transforms._pca`.

Covers three layers of the PCA transform:

* the runtime ``PCATransform`` API on plain numpy arrays (original tests,
  unchanged),
* the portable ``PCATransformConfig`` (``build`` / ``build_and_fit`` /
  round-trip through ``to_spec`` / ``from_spec`` — the object that ends up in
  ``ModelConfig``), and
* the AnnData adapter on ``BaseTransform`` (``fit_adata`` /
  ``transform_adata`` / ``fit_transform_adata``).
"""

import numpy as np
import pytest
import scanpy as sc
from anndata import AnnData
from pydantic import ValidationError

from flowrun.preprocessing.transforms import PCATransform, PCATransformConfig

# --------------------------- numpy runtime API ---------------------------- #


class TestPCATransform:
    """Tests for PCA dimensionality reduction."""

    def test_fit_transform_basic(self):
        """Test basic fit and transform."""
        rng = np.random.default_rng(42)
        X = rng.standard_normal((100, 50))

        pca = PCATransform(n_components=10)
        X_transformed = pca.fit_transform(X)

        # Check output shape
        assert X_transformed.shape == (100, 10)
        assert pca.is_fitted

    def test_fitted_parameters(self):
        """Test that parameters are correctly stored."""
        rng = np.random.default_rng(42)
        X = rng.standard_normal((50, 20))

        pca = PCATransform(n_components=5)
        pca.fit(X)

        # Check stored parameters
        assert pca.is_fitted
        assert pca.params.n_features == 20
        assert pca.params.n_components == 5
        assert pca.params.mean.shape == (20,)
        assert pca.params.components.shape == (5, 20)
        assert pca.params.explained_variance.shape == (5,)

    def test_transform_with_new_data(self):
        """Test transform on new data."""
        rng = np.random.default_rng(42)
        X_train = rng.standard_normal((100, 30))
        X_test = rng.standard_normal((20, 30))

        pca = PCATransform(n_components=10)
        pca.fit(X_train)
        X_test_transformed = pca.transform(X_test)

        assert X_test_transformed.shape == (20, 10)

    def test_inverse_transform(self):
        """Test inverse transform (lossy reconstruction)."""
        rng = np.random.default_rng(42)
        X = rng.standard_normal((50, 20))

        pca = PCATransform(n_components=10)
        X_transformed = pca.fit_transform(X)
        X_reconstructed = pca.inverse_transform(X_transformed)

        # Shape should match original
        assert X_reconstructed.shape == X.shape

        # Reconstruction should be approximate (lossy)
        error = np.mean((X - X_reconstructed) ** 2)
        assert error > 0  # Some information lost
        assert error < 1.0  # But not too much

    def test_full_components_lossless(self):
        """Test that keeping all components gives lossless reconstruction."""
        rng = np.random.default_rng(42)
        X = rng.standard_normal((50, 20))

        pca = PCATransform(n_components=20)  # Keep all
        X_transformed = pca.fit_transform(X)
        X_reconstructed = pca.inverse_transform(X_transformed)

        # Should be nearly lossless
        assert np.allclose(X, X_reconstructed, atol=1e-6)

    def test_explained_variance_ratio(self):
        """Test explained variance ratio."""
        rng = np.random.default_rng(42)
        X = rng.standard_normal((100, 30))

        pca = PCATransform(n_components=10)
        pca.fit(X)

        var_ratio = pca.explained_variance_ratio()

        # Check properties
        assert len(var_ratio) == 10
        assert np.all(var_ratio >= 0)
        assert np.all(var_ratio <= 1)
        assert np.isclose(var_ratio.sum(), 1.0)
        # First PC should explain most variance
        assert var_ratio[0] == var_ratio.max()

    def test_not_fitted_error(self):
        """Test error when transform called before fit."""
        rng = np.random.default_rng(0)
        pca = PCATransform(n_components=5)
        X = rng.standard_normal((10, 20))

        with pytest.raises(RuntimeError, match="not been fitted"):
            pca.transform(X)

    def test_wrong_features_error(self):
        """Test error when n_features doesn't match."""
        rng = np.random.default_rng(0)
        X_train = rng.standard_normal((50, 30))
        X_test = rng.standard_normal((20, 40))

        pca = PCATransform(n_components=10)
        pca.fit(X_train)

        with pytest.raises(ValueError, match="features"):
            pca.transform(X_test)

    def test_manual_vs_ppca_transform(self):
        """Test manual PCA computation matches PCATransform output."""
        rng = np.random.default_rng(42)
        X = rng.standard_normal((100, 50))

        # Manual PCA
        X_mean = np.array(np.mean(X, axis=0))
        X_centered = np.array(X - X_mean)
        X_pca, _, _, _ = sc.pp.pca(X_centered, zero_center=False, return_info=True, n_comps=30)

        # Using PCATransform
        pca = PCATransform(n_components=30)
        X_ppca_transformed = pca.fit_transform(X)

        # Compare
        assert np.allclose(X_pca, X_ppca_transformed, atol=1e-6)


# --------------------------- config: portable API ------------------------- #


class TestPCATransformConfig:
    """Tests for the portable, serializable :class:`PCATransformConfig`."""

    def test_build_returns_fresh_unfitted_transform(self):
        cfg = PCATransformConfig(n_components=5)
        transform = cfg.build()

        assert isinstance(transform, PCATransform)
        assert not transform.is_fitted
        assert transform.n_components == 5

    def test_build_matches_constructor_numerically(self):
        """``build()`` must produce a transform that behaves like the constructor."""
        rng = np.random.default_rng(0)
        X = rng.standard_normal((40, 15))

        via_config = PCATransformConfig(n_components=5).build().fit_transform(X)
        via_ctor = PCATransform(n_components=5).fit_transform(X)

        np.testing.assert_allclose(via_config, via_ctor, atol=1e-12)

    def test_n_components_none_keeps_all_features(self):
        """``n_components=None`` is the default — keep every component."""
        rng = np.random.default_rng(1)
        X = rng.standard_normal((30, 8))

        transform = PCATransformConfig().build().fit(X)
        assert transform.params.n_components == 8

    def test_config_round_trips(self):
        cfg = PCATransformConfig(n_components=10)
        rebuilt = PCATransformConfig.from_spec(cfg.to_spec())
        assert rebuilt == cfg

    def test_config_round_trips_with_none(self):
        cfg = PCATransformConfig(n_components=None)
        rebuilt = PCATransformConfig.from_spec(cfg.to_spec())
        assert rebuilt.n_components is None

    def test_config_is_registered(self):
        """The registry name is the ``type`` discriminator in the spec."""
        assert PCATransformConfig().to_spec().get("type") == "transform.pca"

    def test_config_rejects_non_integer_n_components(self):
        with pytest.raises(ValidationError):
            PCATransformConfig(n_components=3.5)  # type: ignore[arg-type]


# --------------------------- AnnData adapter ------------------------------ #


class TestPCATransformAnnData:
    """Tests for the ``*_adata`` adapters on :class:`BaseTransform`."""

    @pytest.fixture
    def adata(self) -> AnnData:
        rng = np.random.default_rng(42)
        X = rng.standard_normal((50, 20))
        ad = AnnData(X.copy())
        ad.layers["counts"] = X.copy()
        return ad

    def test_fit_adata_reads_x(self, adata: AnnData):
        transform = PCATransformConfig(n_components=5).build()
        transform.fit_adata(adata)

        assert transform.is_fitted
        assert transform.params.n_features == adata.n_vars

    def test_fit_adata_reads_layer(self, adata: AnnData):
        transform = PCATransformConfig(n_components=5).build()
        transform.fit_adata(adata, layer="counts")

        # Should match a fit on the raw layer values.
        expected = PCATransform(n_components=5).fit(adata.layers["counts"])
        np.testing.assert_allclose(transform.params.mean, expected.params.mean)

    def test_transform_adata_writes_to_out_obsm(self, adata: AnnData):
        transform = PCATransformConfig(n_components=5).build_and_fit(adata)
        original_X = adata.X.copy()

        transform.transform_adata(adata, out_obsm="X_pca", inplace=True)

        assert "X_pca" in adata.obsm
        assert adata.obsm["X_pca"].shape == (adata.n_obs, 5)
        # Source X untouched — the output went to obsm.
        np.testing.assert_array_equal(adata.X, original_X)

    def test_transform_adata_layer_to_obsm(self, adata: AnnData):
        transform = PCATransformConfig(n_components=5).build_and_fit(adata, layer="counts")
        transform.transform_adata(adata, layer="counts", out_obsm="X_pca", inplace=True)

        assert adata.obsm["X_pca"].shape == (adata.n_obs, 5)
        np.testing.assert_array_equal(adata.layers["counts"], adata.X)

    def test_transform_adata_inplace_false_returns_copy(self, adata: AnnData):
        original_X = adata.X.copy()
        transform = PCATransformConfig(n_components=5).build_and_fit(adata)

        out = transform.transform_adata(adata, out_obsm="X_pca", inplace=False)

        assert out is not adata
        assert "X_pca" in out.obsm
        assert "X_pca" not in adata.obsm
        np.testing.assert_array_equal(adata.X, original_X)

    def test_build_and_fit_returns_fitted_transform(self, adata: AnnData):
        transform = PCATransformConfig(n_components=5).build_and_fit(adata)

        assert isinstance(transform, PCATransform)
        assert transform.is_fitted
        assert transform.params.n_components == 5

    def test_fit_transform_adata_matches_two_step(self, adata: AnnData):
        """``fit_transform_adata`` == ``fit_adata`` + ``transform_adata``."""
        stepwise = PCATransformConfig(n_components=5).build()
        stepwise.fit_adata(adata)
        stepwise.transform_adata(adata, out_obsm="stepwise", inplace=True)

        one_shot = PCATransformConfig(n_components=5).build()
        one_shot.fit_transform_adata(adata, out_obsm="one_shot", inplace=True)

        np.testing.assert_allclose(adata.obsm["stepwise"], adata.obsm["one_shot"], atol=1e-10)


# --------------------------- parameter access ----------------------------- #


class TestParameterAccess:
    def test_pca_parameter_access(self):
        """Test accessing PCA parameters."""
        rng = np.random.default_rng(42)
        X = rng.standard_normal((50, 20))

        pca = PCATransform(n_components=5)
        pca.fit(X)

        # Access via params
        assert pca.params.mean is not None
        assert pca.params.components is not None
        assert pca.params.explained_variance is not None
        assert pca.params.n_components == 5
        assert pca.params.n_features == 20
