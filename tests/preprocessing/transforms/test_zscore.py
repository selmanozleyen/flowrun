"""Tests for :mod:`flowrun.preprocessing.transforms._zscore`.

Covers three layers of the Z-score transform:

* the runtime ``ZScoreTransform`` API on plain numpy arrays (original tests,
  unchanged),
* the portable ``ZScoreTransformConfig`` (``build`` / ``build_and_fit`` /
  round-trip through ``to_spec`` / ``from_spec`` — the object that ends up in
  ``ModelConfig``), and
* the AnnData adapter on ``BaseTransform`` (``fit_adata`` /
  ``transform_adata`` / ``fit_transform_adata``).
"""

import numpy as np
import pytest
from anndata import AnnData
from pydantic import ValidationError
from scipy.sparse import csr_matrix

from flowrun.preprocessing.transforms import ZScoreTransform, ZScoreTransformConfig

# --------------------------- numpy runtime API ---------------------------- #


class TestZScoreTransform:
    """Tests for Z-score normalization."""

    def test_fit_transform_basic(self):
        """Test basic fit and transform."""
        X = np.array([[1, 10], [2, 20], [3, 30], [4, 40], [5, 50]], dtype=float)

        zscore = ZScoreTransform()
        X_transformed = zscore.fit_transform(X)

        # Check transformed data has mean ~0 and std ~1
        assert np.allclose(X_transformed.mean(axis=0), 0, atol=1e-10)
        assert np.allclose(X_transformed.std(axis=0), 1, atol=1e-10)

    def test_fitted_parameters(self):
        """Test that parameters are correctly stored."""
        X = np.array([[1, 10], [2, 20], [3, 30]], dtype=float)

        zscore = ZScoreTransform()
        zscore.fit(X)

        # Check stored parameters
        assert zscore.is_fitted
        assert zscore.params.n_features == 2
        assert np.allclose(zscore.params.mean, [2.0, 20.0])
        assert len(zscore.params.std) == 2

    def test_transform_with_new_data(self):
        """Test transform on new data using fitted parameters."""
        X_train = np.array([[1, 10], [2, 20], [3, 30]], dtype=float)
        X_test = np.array([[2, 20], [4, 40]], dtype=float)

        zscore = ZScoreTransform()
        zscore.fit(X_train)
        X_test_transformed = zscore.transform(X_test)

        # Check shape preserved
        assert X_test_transformed.shape == X_test.shape

        # Test specific values
        mean = X_train.mean(axis=0)
        std = X_train.std(axis=0)
        expected = (X_test - mean) / std
        assert np.allclose(X_test_transformed, expected)

    def test_inverse_transform(self):
        """Test that inverse transform recovers original data."""
        X = np.array([[1, 10], [2, 20], [3, 30], [4, 40]], dtype=float)

        zscore = ZScoreTransform()
        X_transformed = zscore.fit_transform(X)
        X_reconstructed = zscore.inverse_transform(X_transformed)

        # Should recover original data exactly
        assert np.allclose(X, X_reconstructed)

    def test_not_fitted_error(self):
        """Test error when transform called before fit."""
        zscore = ZScoreTransform()
        X = np.array([[1, 2], [3, 4]])

        with pytest.raises(RuntimeError, match="not been fitted"):
            zscore.transform(X)

    def test_wrong_features_error(self):
        """Test error when n_features doesn't match."""
        X_train = np.array([[1, 2, 3]], dtype=float)
        X_test = np.array([[1, 2]], dtype=float)

        zscore = ZScoreTransform()
        zscore.fit(X_train)

        with pytest.raises(ValueError, match="features"):
            zscore.transform(X_test)

    def test_epsilon_prevents_division_by_zero(self):
        """Test that epsilon prevents division by zero for constant features."""
        X = np.array([[1, 5], [1, 10], [1, 15]], dtype=float)  # First column constant

        zscore = ZScoreTransform(epsilon=1e-8)
        X_transformed = zscore.fit_transform(X)

        # Should not raise error or produce NaN/Inf
        assert not np.any(np.isnan(X_transformed))
        assert not np.any(np.isinf(X_transformed))

    def test_sparse_input(self):
        """Test that sparse matrices are converted properly."""
        X_dense = np.array([[1, 0, 2], [0, 3, 0], [4, 0, 5]], dtype=float)
        X_sparse = csr_matrix(X_dense)

        zscore = ZScoreTransform()
        X_transformed = zscore.fit_transform(X_sparse.toarray())

        assert X_transformed.shape == X_dense.shape


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_single_sample(self):
        """Test with single sample."""
        X = np.array([[1, 2, 3]], dtype=float)

        zscore = ZScoreTransform()
        X_transformed = zscore.fit_transform(X)

        # Should not crash
        assert X_transformed.shape == (1, 3)

    def test_single_feature(self):
        """Test with single feature."""
        X = np.array([[1], [2], [3], [4]], dtype=float)

        zscore = ZScoreTransform()
        X_transformed = zscore.fit_transform(X)

        assert X_transformed.shape == (4, 1)

    def test_invalid_input_not_2d(self):
        """Test error for non-2D input."""
        X = np.array([1, 2, 3, 4])

        zscore = ZScoreTransform()

        with pytest.raises(ValueError, match="2D array"):
            zscore.fit(X)

    def test_invalid_input_type(self):
        """Test error for non-numpy input."""
        X = [[1, 2], [3, 4]]  # List, not numpy array

        zscore = ZScoreTransform()

        with pytest.raises(TypeError, match="numpy array"):
            zscore.fit(X)

    def test_empty_array(self):
        """Test error for empty array."""
        X = np.array([]).reshape(0, 5)

        zscore = ZScoreTransform()

        with pytest.raises(ValueError, match="at least one sample"):
            zscore.fit(X)


# --------------------------- config: portable API ------------------------- #


class TestZScoreTransformConfig:
    """Tests for the portable, serializable :class:`ZScoreTransformConfig`."""

    def test_build_returns_fresh_unfitted_transform(self):
        cfg = ZScoreTransformConfig(epsilon=1e-6)
        transform = cfg.build()

        assert isinstance(transform, ZScoreTransform)
        assert not transform.is_fitted
        # Runtime ``epsilon`` is stored on the private attribute.
        assert transform._epsilon == pytest.approx(1e-6)

    def test_build_matches_constructor_numerically(self):
        """``build()`` must produce a transform that behaves like the constructor."""
        rng = np.random.default_rng(0)
        X = rng.standard_normal((30, 8)) * np.arange(1, 9)

        via_config = ZScoreTransformConfig(epsilon=1e-8).build().fit_transform(X)
        via_ctor = ZScoreTransform(epsilon=1e-8).fit_transform(X)

        np.testing.assert_allclose(via_config, via_ctor, atol=1e-12)

    def test_default_epsilon(self):
        """The default epsilon is the documented ``1e-8``."""
        assert ZScoreTransformConfig().epsilon == pytest.approx(1e-8)

    def test_config_round_trips(self):
        cfg = ZScoreTransformConfig(epsilon=1e-6)
        rebuilt = ZScoreTransformConfig.from_spec(cfg.to_spec())
        assert rebuilt == cfg

    def test_config_round_trips_with_default(self):
        cfg = ZScoreTransformConfig()
        rebuilt = ZScoreTransformConfig.from_spec(cfg.to_spec())
        assert rebuilt.epsilon == pytest.approx(cfg.epsilon)

    def test_config_is_registered(self):
        """The registry name is the ``type`` discriminator in the spec."""
        assert ZScoreTransformConfig().to_spec().get("type") == "transform.zscore"

    def test_config_rejects_negative_epsilon(self):
        """A negative epsilon would defeat the numerical-stability guard."""
        with pytest.raises(ValidationError):
            ZScoreTransformConfig(epsilon=-1.0)


# --------------------------- AnnData adapter ------------------------------ #
class TestZScoreTransformAnnData:
    """Tests for the ``*_adata`` adapters on :class:`BaseTransform`."""

    @pytest.fixture
    def adata(self) -> AnnData:
        rng = np.random.default_rng(42)
        X = rng.standard_normal((50, 20)) * np.arange(1, 21)
        ad = AnnData(X.copy())
        ad.layers["counts"] = X.copy()
        return ad

    def test_fit_adata_reads_x(self, adata: AnnData):
        transform = ZScoreTransformConfig().build()
        transform.fit_adata(adata)

        assert transform.is_fitted
        assert transform.params.n_features == adata.n_vars
        np.testing.assert_allclose(transform.params.mean, adata.X.mean(axis=0), atol=1e-12)

    def test_fit_adata_reads_layer(self, adata: AnnData):
        transform = ZScoreTransformConfig().build()
        transform.fit_adata(adata, layer="counts")

        np.testing.assert_allclose(transform.params.mean, adata.layers["counts"].mean(axis=0), atol=1e-12)

    def test_transform_adata_inplace_overwrites_x(self, adata: AnnData):
        transform = ZScoreTransformConfig().build_and_fit(adata)
        transform.transform_adata(adata, inplace=True)

        # X is now z-scored.
        np.testing.assert_allclose(adata.X.mean(axis=0), 0.0, atol=1e-8)
        np.testing.assert_allclose(adata.X.std(axis=0), 1.0, atol=1e-6)

    def test_transform_adata_layer_to_layer(self, adata: AnnData):
        original_X = adata.X.copy()
        transform = ZScoreTransformConfig().build_and_fit(adata, layer="counts")
        transform.transform_adata(adata, layer="counts", out_layer="counts_z", inplace=True)

        assert "counts_z" in adata.layers
        assert adata.layers["counts_z"].shape == adata.layers["counts"].shape
        # Source X untouched.
        np.testing.assert_array_equal(adata.X, original_X)

    def test_transform_adata_inplace_false_returns_copy(self, adata: AnnData):
        original_X = adata.X.copy()
        transform = ZScoreTransformConfig().build_and_fit(adata)

        out = transform.transform_adata(adata, inplace=False)

        assert out is not adata
        assert not np.array_equal(out.X, original_X)
        np.testing.assert_array_equal(adata.X, original_X)

    def test_inverse_transform_adata_recovers_x(self, adata: AnnData):
        original_X = adata.X.copy()
        transform = ZScoreTransformConfig().build_and_fit(adata)

        transform.transform_adata(adata, inplace=True)
        transform.inverse_transform_adata(adata, inplace=True)

        np.testing.assert_allclose(adata.X, original_X, atol=1e-6)

    def test_build_and_fit_returns_fitted_transform(self, adata: AnnData):
        transform = ZScoreTransformConfig(epsilon=1e-6).build_and_fit(adata)

        assert isinstance(transform, ZScoreTransform)
        assert transform.is_fitted
        assert transform._epsilon == pytest.approx(1e-6)

    def test_fit_transform_adata_matches_two_step(self, adata: AnnData):
        """``fit_transform_adata`` == ``fit_adata`` + ``transform_adata``."""
        stepwise = ZScoreTransformConfig().build()
        stepwise.fit_adata(adata)
        stepwise.transform_adata(adata, out_layer="stepwise", inplace=True)

        one_shot = ZScoreTransformConfig().build()
        one_shot.fit_transform_adata(adata, out_layer="one_shot", inplace=True)

        np.testing.assert_allclose(adata.layers["stepwise"], adata.layers["one_shot"], atol=1e-12)


# --------------------------- parameter access ----------------------------- #


class TestParameterAccess:
    """Test accessing fitted parameters."""

    def test_zscore_parameter_access(self):
        """Test accessing Z-score parameters."""
        X = np.array([[1, 10], [2, 20], [3, 30]], dtype=float)

        zscore = ZScoreTransform()
        zscore.fit(X)

        # Access via params
        assert zscore.params.mean is not None
        assert zscore.params.std is not None
        assert zscore.params.n_features == 2

        # Check values
        assert np.allclose(zscore.params.mean, [2.0, 20.0])
