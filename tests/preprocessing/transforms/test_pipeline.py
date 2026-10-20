"""Tests for chaining preprocessors (Z-score / PCA / Whitening).

The same three pipelines are exercised at each layer of the new transform
stack so that regressions surface at the layer where they live:

* the runtime ``BaseTransform`` API on plain numpy arrays,
* the portable ``TransformConfig`` API (``build`` / ``build_and_fit``), and
* the AnnData adapter (``fit_adata`` / ``transform_adata``).

A final test checks that every ``TransformConfig`` round-trips through
``to_spec`` / ``from_spec`` — the contract that keeps a ``ModelConfig`` written
by ``Model.save_pretrained`` portable.
"""

import numpy as np
import pytest
from anndata import AnnData

from flowrun.preprocessing.transforms import (
    PCATransform,
    PCATransformConfig,
    TransformConfig,
    WhiteningTransform,
    WhiteningTransformConfig,
    ZScoreTransform,
    ZScoreTransformConfig,
)

# --------------------------- shared fixtures ------------------------------ #

N_SAMPLES = 100
N_FEATURES = 50


@pytest.fixture
def X_train() -> np.ndarray:
    """Anisotropic training data — different scale on every feature."""
    rng = np.random.default_rng(42)
    return rng.standard_normal((N_SAMPLES, N_FEATURES)) * np.arange(1, N_FEATURES + 1)


@pytest.fixture
def X_test() -> np.ndarray:
    rng = np.random.default_rng(7)
    return rng.standard_normal((20, N_FEATURES)) * np.arange(1, N_FEATURES + 1)


@pytest.fixture
def adata_train(X_train: np.ndarray) -> AnnData:
    adata = AnnData(X_train.copy())
    adata.layers["counts"] = X_train.copy()
    return adata


# --------------------------- numpy: runtime API --------------------------- #


class TestPreprocessingPipeline:
    """Chains built from raw ``BaseTransform`` instances on numpy arrays."""

    def test_zscore_then_pca(self, X_train: np.ndarray, X_test: np.ndarray) -> None:
        """Test Z-score followed by PCA."""
        zscore = ZScoreTransform()
        X_train_norm = zscore.fit_transform(X_train)
        X_test_norm = zscore.transform(X_test)

        pca = PCATransform(n_components=10)
        X_train_pca = pca.fit_transform(X_train_norm)
        X_test_pca = pca.transform(X_test_norm)

        assert X_train_pca.shape == (N_SAMPLES, 10)
        assert X_test_pca.shape == (20, 10)

        # Inverse pipeline returns to the original shape.
        X_train_recon_norm = pca.inverse_transform(X_train_pca)
        X_train_recon = zscore.inverse_transform(X_train_recon_norm)
        assert X_train_recon.shape == X_train.shape

    def test_zscore_then_whitening(self, X_train: np.ndarray) -> None:
        """Z-score followed by ZCA whitening produces unit covariance."""
        zscore = ZScoreTransform()
        whitening = WhiteningTransform(method="zca")

        X_norm = zscore.fit_transform(X_train)
        X_white = whitening.fit_transform(X_norm)

        assert X_white.shape == X_train.shape
        cov = np.atleast_2d(np.cov(X_white, rowvar=False))
        np.testing.assert_allclose(cov, np.eye(N_FEATURES), atol=1e-4)

    def test_zscore_pca_whitening_chain(self, X_train: np.ndarray) -> None:
        """Full chain: Z-score → PCA (drop to 10D) → whitening (rank 10)."""
        zscore = ZScoreTransform()
        pca = PCATransform(n_components=10)
        whitening = WhiteningTransform(method="zca")

        X = zscore.fit_transform(X_train)
        X = pca.fit_transform(X)
        X = whitening.fit_transform(X)

        assert X.shape == (N_SAMPLES, 10)
        cov = np.atleast_2d(np.cov(X, rowvar=False))
        np.testing.assert_allclose(cov, np.eye(10), atol=1e-4)

    def test_fit_transform_populates_params(self, X_train: np.ndarray) -> None:
        """``fit_transform`` returns data *and* leaves ``params`` populated."""
        pca = PCATransform(n_components=5)
        X_transformed = pca.fit_transform(X_train)

        assert X_transformed.shape == (N_SAMPLES, 5)
        assert pca.params.is_fitted
        assert pca.params.n_components == 5


# --------------------------- config: portable API ------------------------- #


class TestConfigPipeline:
    """Chains built via ``TransformConfig.build`` and ``build_and_fit``.

    These mirror :class:`TestPreprocessingPipeline` — same chains, same
    assertions — but go through the config layer that ends up in ``ModelConfig``.
    """

    def test_config_chain_zscore_then_pca(self, X_train: np.ndarray, X_test: np.ndarray) -> None:
        zscore_cfg = ZScoreTransformConfig(epsilon=1e-8)
        pca_cfg = PCATransformConfig(n_components=10)

        zscore = zscore_cfg.build()
        pca = pca_cfg.build()

        X_train_norm = zscore.fit_transform(X_train)
        X_test_norm = zscore.transform(X_test)
        X_train_pca = pca.fit_transform(X_train_norm)
        X_test_pca = pca.transform(X_test_norm)

        assert X_train_pca.shape == (N_SAMPLES, 10)
        assert X_test_pca.shape == (20, 10)

    def test_config_build_matches_constructor(self, X_train: np.ndarray) -> None:
        """``PCATransformConfig.build()`` produces a numerically identical transform."""
        cfg = PCATransformConfig(n_components=10)
        via_config = cfg.build().fit_transform(X_train)
        via_ctor = PCATransform(n_components=10).fit_transform(X_train)
        np.testing.assert_allclose(via_config, via_ctor, atol=1e-10)

    def test_config_chain_inverse(self, X_train: np.ndarray) -> None:
        zscore = ZScoreTransformConfig().build()
        pca = PCATransformConfig(n_components=10).build()
        whitening = WhiteningTransformConfig(method="zca").build()

        X = zscore.fit_transform(X_train)
        X = pca.fit_transform(X)
        X = whitening.fit_transform(X)
        X = whitening.inverse_transform(X)
        X = pca.inverse_transform(X)
        X = zscore.inverse_transform(X)

        # End-to-end reconstruction is lossy only because of the PCA step,
        # so it matches the PCA-only reconstruction to numerical precision.
        pca_only = PCATransform(n_components=10)
        X_pca = pca_only.fit_transform(ZScoreTransform().fit_transform(X_train))
        X_pca_recon = ZScoreTransform().fit(X_train).inverse_transform(pca_only.inverse_transform(X_pca))
        assert np.mean((X - X_pca_recon) ** 2) < 1e-20


# --------------------------- AnnData adapters ----------------------------- #


class TestAnnDataPipeline:
    """Chains driven by the ``*_adata`` adapters on ``BaseTransform``."""

    def test_zscore_inplace_then_pca_to_obsm(self, adata_train: AnnData, X_train: np.ndarray) -> None:
        """Z-score overwrites ``.X``; PCA reads ``.X`` and writes ``.obsm``."""
        zscore = ZScoreTransformConfig().build()
        pca = PCATransformConfig(n_components=10).build()

        # Stage 1 — normalise X in place.
        zscore.fit_adata(adata_train)
        zscore.transform_adata(adata_train, inplace=True)

        # Stage 2 — reduce to PC space, don't touch X.
        pca.fit_adata(adata_train)
        pca.transform_adata(adata_train, out_obsm="X_pca", inplace=True)

        assert "X_pca" in adata_train.obsm
        assert adata_train.obsm["X_pca"].shape == (adata_train.n_obs, 10)
        # X still holds the *normalised* data (mean ≈ 0, std ≈ 1).
        np.testing.assert_allclose(adata_train.X.mean(axis=0), 0.0, atol=1e-6)

    def test_layer_to_obsm(self, adata_train: AnnData) -> None:
        """Read from ``.layers['counts']`` and write to ``.obsm['X_white']``."""
        whitening = WhiteningTransformConfig(method="zca").build_and_fit(adata_train, layer="counts")
        whitening.transform_adata(
            adata_train,
            layer="counts",
            out_obsm="X_white",
            inplace=True,
        )

        assert "X_white" in adata_train.obsm
        assert adata_train.obsm["X_white"].shape == (
            adata_train.n_obs,
            adata_train.n_vars,
        )
        # Source layer untouched.
        np.testing.assert_array_equal(adata_train.layers["counts"], adata_train.X)

    def test_inplace_false_leaves_original_untouched(self, adata_train: AnnData) -> None:
        original_X = adata_train.X.copy()

        zscore = ZScoreTransformConfig().build_and_fit(adata_train)
        out = zscore.transform_adata(adata_train, inplace=False)

        assert out is not adata_train
        np.testing.assert_array_equal(adata_train.X, original_X)
        assert not np.array_equal(out.X, original_X)

    def test_build_and_fit_one_shot(self, adata_train: AnnData) -> None:
        """``build_and_fit`` returns a fitted runtime transform, not a config."""
        xf = PCATransformConfig(n_components=10).build_and_fit(adata_train)
        assert isinstance(xf, PCATransform)
        assert xf.is_fitted
        assert xf.params.n_components == 10

    def test_full_chain_with_layer_and_obsm(self, adata_train: AnnData) -> None:
        """Z-score X → PCA on X → whitening on obsm, all via AnnData."""
        zscore = ZScoreTransformConfig().build_and_fit(adata_train)
        zscore.transform_adata(adata_train, inplace=True)

        pca = PCATransformConfig(n_components=10).build_and_fit(adata_train)
        pca.transform_adata(adata_train, out_obsm="X_pca", inplace=True)

        whitening = WhiteningTransformConfig(method="zca").build_and_fit(adata_train, obsm="X_pca")
        whitening.transform_adata(
            adata_train,
            obsm="X_pca",
            out_obsm="X_pca_white",
            inplace=True,
        )

        assert "X_pca_white" in adata_train.obsm
        cov = np.atleast_2d(np.cov(adata_train.obsm["X_pca_white"], rowvar=False))
        np.testing.assert_allclose(cov, np.eye(10), atol=1e-4)


# --------------------------- portability ---------------------------------- #


class TestSerializability:
    """Every ``TransformConfig`` must round-trip through ``to_spec``/``from_spec``.

    This is what ``Model.save_pretrained`` writes into ``config.json``; a config that cannot
    round-trip makes a trained run unloadable.
    """

    @pytest.mark.parametrize(
        "config",
        [
            ZScoreTransformConfig(epsilon=1e-8),
            PCATransformConfig(n_components=10),
            PCATransformConfig(n_components=None),
            WhiteningTransformConfig(method="zca", epsilon=1e-5),
            WhiteningTransformConfig(method="pca"),
            WhiteningTransformConfig(method="cholesky"),
        ],
    )
    def test_transform_config_round_trips(self, config: TransformConfig) -> None:
        rebuilt = type(config).from_spec(config.to_spec())
        assert rebuilt == config

    def test_pipeline_of_configs_round_trips(self) -> None:
        """A tuple of configs — the shape a spec actually stores — round-trips."""
        configs = {
            "pred": ZScoreTransformConfig(epsilon=1e-8),
            "target": PCATransformConfig(n_components=10),
            "whitening": WhiteningTransformConfig(method="zca"),
        }
        for name, cfg in configs.items():
            rebuilt = type(cfg).from_spec(cfg.to_spec())
            assert rebuilt == cfg, name
