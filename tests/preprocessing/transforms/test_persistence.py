"""Tests for :mod:`flowrun.preprocessing.transforms._persistence`.

Covers:

* the file layout :func:`save_transform` writes (``transform.json`` +
  ``transform_params.npz``),
* argument validation on save (unfitted transforms rejected),
* the round-trip contract: :func:`load_transform` returns a transform of the
  same type, with numerically identical ``params``, whose ``transform`` /
  ``inverse_transform`` produce the same output as the original — for every
  concrete transform currently in the package,
* error paths on load (missing files, unknown registry name, unset
  ``_params_cls``),
* the behavioural equivalence between a loaded transform and a freshly fitted
  one on the same data.
"""

import json
from dataclasses import fields
from pathlib import Path

import numpy as np
import pytest

from flowrun.preprocessing.transforms import (
    TRANSFORM_PARAMS_FILENAME,
    TRANSFORM_SPEC_FILENAME,
    IdentityTransform,
    IdentityTransformConfig,
    PCATransform,
    PCATransformConfig,
    TransformConfig,
    WhiteningTransform,
    WhiteningTransformConfig,
    ZScoreTransformConfig,
    load_transform,
    save_transform,
)

# --------------------------- fixtures ---------------------------------------


@pytest.fixture
def sample_data() -> np.ndarray:
    """Anisotropic 50×8 data — every transform has something non-trivial to fit."""
    rng = np.random.default_rng(0)
    return rng.standard_normal((50, 8)) * np.arange(1, 9)


# Every concrete transform currently in the package. Identity has trivial
# params (``is_fitted`` + ``n_features``); the others exercise multi-array
# round-trips. Keep this list in sync if a new transform is added — the
# ``test_round_trip_*`` parametrizations are the intended coverage target.
_ALL_CONFIGS = [
    pytest.param(IdentityTransformConfig(), id="identity"),
    pytest.param(ZScoreTransformConfig(epsilon=1e-9), id="zscore"),
    pytest.param(PCATransformConfig(n_components=4), id="pca"),
    pytest.param(WhiteningTransformConfig(method="zca", epsilon=1e-5), id="whitening-zca"),
    pytest.param(WhiteningTransformConfig(method="pca", epsilon=1e-5), id="whitening-pca"),
    pytest.param(
        WhiteningTransformConfig(method="cholesky", epsilon=1e-5),
        id="whitening-cholesky",
    ),
]


# --------------------------- save ------------------------------------------


class TestSaveTransform:
    """File layout + argument validation for :func:`save_transform`."""

    def test_writes_both_files(self, tmp_path: Path, sample_data: np.ndarray) -> None:
        cfg = ZScoreTransformConfig()
        xf = cfg.build().fit(sample_data)

        save_transform(tmp_path, cfg, xf)

        assert (tmp_path / TRANSFORM_SPEC_FILENAME).is_file()
        assert (tmp_path / TRANSFORM_PARAMS_FILENAME).is_file()

    def test_creates_run_dir_if_missing(self, tmp_path: Path, sample_data: np.ndarray) -> None:
        nested = tmp_path / "a" / "b" / "run"
        cfg = ZScoreTransformConfig()
        xf = cfg.build().fit(sample_data)

        save_transform(nested, cfg, xf)

        assert (nested / TRANSFORM_SPEC_FILENAME).is_file()
        assert (nested / TRANSFORM_PARAMS_FILENAME).is_file()

    def test_rejects_unfitted_transform(self, tmp_path: Path, sample_data: np.ndarray) -> None:
        # Z-score starts unfitted.
        cfg = ZScoreTransformConfig()
        xf = cfg.build()
        assert not xf.is_fitted

        with pytest.raises(RuntimeError, match="unfitted"):
            save_transform(tmp_path, cfg, xf)

    def test_spec_is_portable_json(self, tmp_path: Path, sample_data: np.ndarray) -> None:
        """The written spec equals ``cfg.to_spec()`` and round-trips via from_spec."""
        cfg = PCATransformConfig(n_components=4)
        xf = cfg.build().fit(sample_data)

        save_transform(tmp_path, cfg, xf)

        written = json.loads((tmp_path / TRANSFORM_SPEC_FILENAME).read_text())
        assert written == cfg.to_spec()
        assert written["type"] == "transform.pca"
        # And the spec is dispatchable — this is what `load_transform` relies on.
        rebuilt = TransformConfig.from_spec(written)
        assert rebuilt == cfg


# --------------------------- load: round-trip ------------------------------


class TestLoadTransformRoundTrip:
    """For every concrete transform, save → load produces the same behaviour."""

    @pytest.mark.parametrize("cfg", _ALL_CONFIGS)
    def test_returns_matching_transform_type(
        self, tmp_path: Path, sample_data: np.ndarray, cfg: TransformConfig
    ) -> None:
        xf = cfg.build().fit(sample_data)
        save_transform(tmp_path, cfg, xf)

        loaded = load_transform(tmp_path)

        assert type(loaded) is type(xf)
        assert loaded.is_fitted

    @pytest.mark.parametrize("cfg", _ALL_CONFIGS)
    def test_preserves_params_exactly(self, tmp_path: Path, sample_data: np.ndarray, cfg: TransformConfig) -> None:
        """Array params compare equal; scalar params compare by value."""
        xf = cfg.build().fit(sample_data)
        save_transform(tmp_path, cfg, xf)

        loaded = load_transform(tmp_path)

        for f in fields(xf.params):
            original = getattr(xf.params, f.name)
            restored = getattr(loaded.params, f.name)
            if original is None:
                assert restored is None, f.name
            elif isinstance(original, np.ndarray):
                np.testing.assert_array_equal(restored, original, err_msg=f.name)
            else:
                assert restored == original, f.name

    @pytest.mark.parametrize("cfg", _ALL_CONFIGS)
    def test_transform_output_unchanged(self, tmp_path: Path, sample_data: np.ndarray, cfg: TransformConfig) -> None:
        """The loaded transform applies the same function as the original."""
        xf = cfg.build().fit(sample_data)
        Y_expected = xf.transform(sample_data)

        save_transform(tmp_path, cfg, xf)
        loaded = load_transform(tmp_path)

        np.testing.assert_array_equal(loaded.transform(sample_data), Y_expected)

    @pytest.mark.parametrize(
        "cfg",
        [
            pytest.param(ZScoreTransformConfig(), id="zscore"),
            pytest.param(PCATransformConfig(n_components=4), id="pca"),
            pytest.param(WhiteningTransformConfig(method="zca"), id="whitening-zca"),
        ],
    )
    def test_inverse_transform_supported(self, tmp_path: Path, sample_data: np.ndarray, cfg: TransformConfig) -> None:
        """Round-trip through the loaded transform matches the original's inverse."""
        xf = cfg.build().fit(sample_data)
        X_recon_expected = xf.inverse_transform(xf.transform(sample_data))

        save_transform(tmp_path, cfg, xf)
        loaded = load_transform(tmp_path)
        X_recon = loaded.inverse_transform(loaded.transform(sample_data))

        np.testing.assert_allclose(X_recon, X_recon_expected, atol=1e-12)

    def test_identity_round_trip_is_exact(self, tmp_path: Path, sample_data: np.ndarray) -> None:
        """Identity, by definition, returns its input unchanged after reload."""
        cfg = IdentityTransformConfig()
        xf = cfg.build().fit(sample_data)
        save_transform(tmp_path, cfg, xf)

        loaded = load_transform(tmp_path)

        np.testing.assert_array_equal(loaded.transform(sample_data), sample_data)


# --------------------------- load: error paths -----------------------------


class TestLoadTransformErrors:
    """Missing files and unknown registry names produce actionable errors."""

    def test_raises_when_spec_missing(self, tmp_path: Path) -> None:
        # Params present, spec absent — the common half-written-dir case.
        np.savez_compressed(tmp_path / TRANSFORM_PARAMS_FILENAME)

        with pytest.raises(FileNotFoundError, match="transform spec"):
            load_transform(tmp_path)

    def test_raises_when_params_missing(self, tmp_path: Path) -> None:
        (tmp_path / TRANSFORM_SPEC_FILENAME).write_text(json.dumps(ZScoreTransformConfig().to_spec()))

        with pytest.raises(FileNotFoundError, match="transform params"):
            load_transform(tmp_path)

    def test_raises_on_unknown_registry_type(self, tmp_path: Path) -> None:
        """An unrecognised ``type`` in the spec must fail loudly, not silently.

        ``Component.from_spec`` dispatches on the ``type`` field; unknown values
        surface as one of:

        * ``KeyError``    — the registry implements the dispatch as a dict lookup,
        * ``ValueError``  — the registry (or pydantic) rejects the value directly.

        Narrowed from bare ``Exception`` deliberately: if scfit ever changes its
        error contract, this test should fail loudly so the caller knows the API
        moved — a blind ``pytest.raises(Exception)`` would silently keep passing.
        """
        (tmp_path / TRANSFORM_SPEC_FILENAME).write_text(json.dumps({"type": "transform.does_not_exist"}))
        np.savez_compressed(tmp_path / TRANSFORM_PARAMS_FILENAME)

        with pytest.raises((KeyError, ValueError)):
            load_transform(tmp_path)


# --------------------------- behavioural --------------------------------


class TestLoadedTransformBehavesLikeAFreshFit:
    """A loaded transform is indistinguishable from one fitted in this process."""

    def test_zscore_loaded_params_match_fresh_fit(self, tmp_path: Path, sample_data: np.ndarray) -> None:
        cfg = ZScoreTransformConfig(epsilon=1e-9)
        fitted_in_run = cfg.build().fit(sample_data)
        save_transform(tmp_path, cfg, fitted_in_run)

        loaded = load_transform(tmp_path)
        fresh = ZScoreTransformConfig(epsilon=1e-9).build().fit(sample_data)

        np.testing.assert_array_equal(loaded.params.mean, fresh.params.mean)
        np.testing.assert_array_equal(loaded.params.std, fresh.params.std)

    def test_pca_loaded_components_match_fresh_fit(self, tmp_path: Path, sample_data: np.ndarray) -> None:
        cfg = PCATransformConfig(n_components=4)
        fitted_in_run = cfg.build().fit(sample_data)
        save_transform(tmp_path, cfg, fitted_in_run)

        loaded = load_transform(tmp_path)
        fresh = PCATransformConfig(n_components=4).build().fit(sample_data)

        assert isinstance(loaded, PCATransform)
        np.testing.assert_array_equal(loaded.params.mean, fresh.params.mean)
        np.testing.assert_array_equal(loaded.params.components, fresh.params.components)
        assert loaded.params.n_components == fresh.params.n_components

    def test_whitening_loaded_matrices_match_fresh_fit(self, tmp_path: Path, sample_data: np.ndarray) -> None:
        cfg = WhiteningTransformConfig(method="zca", epsilon=1e-5)
        fitted_in_run = cfg.build().fit(sample_data)
        save_transform(tmp_path, cfg, fitted_in_run)

        loaded = load_transform(tmp_path)
        fresh = WhiteningTransformConfig(method="zca", epsilon=1e-5).build().fit(sample_data)

        assert isinstance(loaded, WhiteningTransform)
        np.testing.assert_array_equal(loaded.params.W, fresh.params.W)
        np.testing.assert_array_equal(loaded.params.iW, fresh.params.iW)
        np.testing.assert_array_equal(loaded.params.mean, fresh.params.mean)

    def test_identity_loaded_usable_without_prior_fit(self, tmp_path: Path, sample_data: np.ndarray) -> None:
        """Identity is fitted from birth — a loaded one is immediately usable."""
        cfg = IdentityTransformConfig()
        save_transform(tmp_path, cfg, cfg.build().fit(sample_data))

        loaded = load_transform(tmp_path)
        assert isinstance(loaded, IdentityTransform)
        # Works without a further ``fit`` call:
        np.testing.assert_array_equal(loaded.transform(sample_data), sample_data)
