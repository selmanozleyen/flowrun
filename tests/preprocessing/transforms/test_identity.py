"""Tests for :mod:`flowrun.preprocessing.transforms._identity`.

Three layers are covered:

* the runtime ``IdentityTransform`` API — always fitted, works without a
  prior ``fit``, accepts any width, returns a copy (no aliasing),
* the portable ``IdentityTransformConfig`` — ``build()`` returns a
  ready-to-use transform, spec round-trips, ``type`` discriminator is
  registered, and
* the AnnData adapter on :class:`BaseTransform` — fit on ``X`` / a layer,
  write to ``obsm`` without aliasing the input.
"""

import numpy as np
import pytest
from anndata import AnnData

from flowrun.preprocessing.transforms import (
    IdentityTransform,
    IdentityTransformConfig,
    ZScoreTransformConfig,
)

# --------------------------- numpy runtime API ---------------------------- #


class TestIdentityTransform:
    """Tests for the no-op transform on plain numpy arrays."""

    def test_fit_transform_returns_input_values(self) -> None:
        X = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

        out = IdentityTransform().fit_transform(X)

        np.testing.assert_array_equal(out, X)

    def test_fit_stores_n_features(self) -> None:
        """``fit`` is optional but records ``n_features`` when called."""
        X = np.random.default_rng(0).standard_normal((10, 7))

        xf = IdentityTransform().fit(X)

        assert xf.is_fitted
        assert xf.params.n_features == 7

    def test_transform_works_without_fit(self) -> None:
        """Identity is a no-op — it must not require a prior ``fit``.

        This is the whole point of the refactor: identity is fitted from
        birth, so ``build()`` alone yields a ready-to-use transform.
        """
        X = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

        Y = IdentityTransform().transform(X)

        np.testing.assert_array_equal(Y, X)

    def test_inverse_transform_works_without_fit(self) -> None:
        X = np.array([[1.0, 2.0, 3.0]])

        Y = IdentityTransform().inverse_transform(X)

        np.testing.assert_array_equal(Y, X)

    def test_transform_returns_copy_not_input(self) -> None:
        """Mutating the output must not mutate the input — no aliasing."""
        X = np.array([[1.0, 2.0, 3.0]])
        xf = IdentityTransform()

        Y = xf.transform(X)

        assert Y is not X
        Y[0, 0] = 999.0
        assert X[0, 0] == 1.0

    def test_inverse_transform_returns_copy_not_input(self) -> None:
        X = np.array([[1.0, 2.0, 3.0]])
        xf = IdentityTransform()

        Y = xf.inverse_transform(X)

        assert Y is not X
        np.testing.assert_array_equal(Y, X)

    def test_inverse_transform_is_identity(self) -> None:
        """``inverse_transform(transform(X)) == X`` exactly."""
        rng = np.random.default_rng(1)
        X = rng.standard_normal((20, 6))

        xf = IdentityTransform()
        np.testing.assert_array_equal(xf.inverse_transform(xf.transform(X)), X)

    def test_transform_preserves_dtype(self) -> None:
        for dtype in (np.float32, np.float64, np.int64):
            X = np.ones((3, 4), dtype=dtype)
            xf = IdentityTransform()
            assert xf.transform(X).dtype == X.dtype

    def test_transform_accepts_any_width(self) -> None:
        """Identity is defined on every width — a fit doesn't constrain it.

        Unlike z-score / PCA / whitening, there is no learned feature-space
        mapping to mismatch against, so ``_validate_n_features`` is not
        applied.
        """
        xf = IdentityTransform().fit(np.zeros((4, 5)))
        Y = xf.transform(np.ones((3, 8)))

        assert Y.shape == (3, 8)
        np.testing.assert_array_equal(Y, np.ones((3, 8)))

    def test_invalid_input_not_2d(self) -> None:
        with pytest.raises(ValueError, match="2D array"):
            IdentityTransform().fit(np.array([1.0, 2.0, 3.0]))

    def test_invalid_input_type(self) -> None:
        with pytest.raises(TypeError, match="numpy array"):
            IdentityTransform().fit([[1.0, 2.0], [3.0, 4.0]])

    def test_invalid_input_not_2d_on_transform(self) -> None:
        """``transform`` still validates the input shape, even without ``fit``."""
        with pytest.raises(ValueError, match="2D array"):
            IdentityTransform().transform(np.array([1.0, 2.0, 3.0]))


# --------------------------- config: portable API ------------------------- #


class TestIdentityTransformConfig:
    """Tests for the portable, serializable :class:`IdentityTransformConfig`."""

    def test_build_returns_ready_to_use_transform(self) -> None:
        """``build()`` returns a transform that is usable immediately."""
        cfg = IdentityTransformConfig()
        transform = cfg.build()

        assert isinstance(transform, IdentityTransform)
        assert transform.is_fitted  # identity is fitted from birth
        # And it actually works without a prior ``fit``.
        np.testing.assert_array_equal(transform.transform(np.ones((2, 3))), np.ones((2, 3)))

    def test_build_matches_constructor(self) -> None:
        """``build()`` is a thin wrapper — same behaviour as the constructor."""
        rng = np.random.default_rng(0)
        X = rng.standard_normal((15, 4))

        via_config = IdentityTransformConfig().build().fit_transform(X)
        via_ctor = IdentityTransform().fit_transform(X)

        np.testing.assert_array_equal(via_config, via_ctor)

    def test_config_round_trips(self) -> None:
        cfg = IdentityTransformConfig()
        rebuilt = IdentityTransformConfig.from_spec(cfg.to_spec())
        assert rebuilt == cfg

    def test_config_is_registered(self) -> None:
        """The registry name is the ``type`` discriminator in the spec."""
        assert IdentityTransformConfig().to_spec().get("type") == "transform.identity"

    def test_config_spec_is_minimal(self) -> None:
        """No hyperparameters — only the type discriminator and the Component version."""
        spec = IdentityTransformConfig().to_spec()
        # ``Component`` injects a ``version`` field into every spec; identity
        # adds nothing else.
        assert set(spec) == {"type"}
        assert spec["type"] == "transform.identity"


# --------------------------- AnnData adapter ------------------------------ #


class TestIdentityTransformAnnData:
    """Tests for the ``*_adata`` adapters on :class:`BaseTransform`."""

    @pytest.fixture
    def adata(self) -> AnnData:
        rng = np.random.default_rng(42)
        X = rng.standard_normal((30, 8)).astype(np.float64)
        ad = AnnData(X.copy())
        ad.layers["counts"] = X.copy()
        return ad

    def test_fit_adata_records_n_features(self, adata: AnnData) -> None:
        xf = IdentityTransformConfig().build()
        xf.fit_adata(adata)

        assert xf.is_fitted
        assert xf.params.n_features == adata.n_vars

    def test_transform_adata_out_obsm_does_not_alias_x(self, adata: AnnData) -> None:
        """``obsm["X_id"]`` must be a distinct array from ``adata.X``.

        If ``transform`` returned ``adata.X`` itself, ``obsm["X_id"]`` would
        alias ``adata.X`` and mutating one would corrupt the other.
        """
        xf = IdentityTransformConfig().build_and_fit(adata)
        xf.transform_adata(adata, out_obsm="X_id", inplace=True)

        assert "X_id" in adata.obsm
        np.testing.assert_array_equal(adata.obsm["X_id"], adata.X)

        adata.obsm["X_id"][0, 0] = 999.0
        assert adata.X[0, 0] != 999.0

    def test_transform_adata_layer_to_obsm(self, adata: AnnData) -> None:
        xf = IdentityTransformConfig().build_and_fit(adata, layer="counts")
        xf.transform_adata(adata, layer="counts", out_obsm="X_id", inplace=True)

        np.testing.assert_array_equal(adata.obsm["X_id"], adata.layers["counts"])
        # Source layer untouched.
        np.testing.assert_array_equal(adata.layers["counts"], adata.X)

    def test_inplace_false_leaves_original_untouched(self, adata: AnnData) -> None:
        original_X = adata.X.copy()
        xf = IdentityTransformConfig().build_and_fit(adata)

        out = xf.transform_adata(adata, out_obsm="X_id", inplace=False)

        assert out is not adata
        assert "X_id" in out.obsm
        assert "X_id" not in adata.obsm
        np.testing.assert_array_equal(adata.X, original_X)

    def test_build_and_fit_returns_fitted_transform(self, adata: AnnData) -> None:
        xf = IdentityTransformConfig().build_and_fit(adata)

        assert isinstance(xf, IdentityTransform)
        assert xf.is_fitted

    def test_fit_transform_adata_matches_two_step(self, adata: AnnData) -> None:
        stepwise = IdentityTransformConfig().build()
        stepwise.fit_adata(adata)
        stepwise.transform_adata(adata, out_obsm="stepwise", inplace=True)

        one_shot = IdentityTransformConfig().build()
        one_shot.fit_transform_adata(adata, out_obsm="one_shot", inplace=True)

        np.testing.assert_array_equal(adata.obsm["stepwise"], adata.obsm["one_shot"])


# --------------------------- pipeline integration ------------------------- #


class TestIdentityInPipeline:
    """Identity's job is to be invisible in a chain."""

    def test_identity_then_zscore_matches_zscore_alone(self) -> None:
        rng = np.random.default_rng(0)
        X = rng.standard_normal((50, 6))

        chained = IdentityTransformConfig().build().fit_transform(X)
        zscore_on_chained = ZScoreTransformConfig().build().fit_transform(chained)
        zscore_direct = ZScoreTransformConfig().build().fit_transform(X)

        np.testing.assert_allclose(zscore_on_chained, zscore_direct, atol=1e-12)

    def test_identity_chain_round_trip(self) -> None:
        """Round-trip through identity is exact and needs no fit."""
        rng = np.random.default_rng(1)
        X = rng.standard_normal((20, 4))

        xf = IdentityTransformConfig().build()
        Y = xf.inverse_transform(xf.transform(X))
        np.testing.assert_array_equal(Y, X)
