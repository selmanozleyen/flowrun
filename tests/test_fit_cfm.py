"""Smoke test: run `flowrun.fit_cfm` against the dummy adata fixture.

Three pair matrices, all built on the same principle: a DM override and the
VF override it implies are **bundled together**, so they cannot drift.

* ``_SCHEMA_PAIRS`` — datamanager schema (categorical_conditions, covariates, paired/
  unpaired, groups) paired with the VF configuration it requires — including
  one condition-encoder input layer per condition key.
* ``_COUPLING_PAIRS`` — coupling method paired with the datamanager reps
  it needs, plus the source encoder for quadratic couplings only.
* ``_AXES`` — orthogonal dimensions on top of ``pair="one_cat_paired"``.

Invariants the pairing tables enforce
-------------------------------------

1. **One condition-encoder input layer per condition key.**
   ``module.condition_encoder`` is a set encoder spec whose
   ``categorical_input_layers`` / ``continuous_input_layers`` are keyed by the
   same names that appear in ``dims.condition_reps_dims`` + ``groups_reps_dims`` /
   ``condition_continuous_dims``; their input widths come from ``dims``. The
   test pairs provide exactly the entries their DM produces, so the
   per-condition input-layer path is exercised end-to-end.
   ``module.condition_encoder`` defaults to ``null`` in the YAML; overrides
   use bare ``path=value`` (not the ``~ +`` pattern) because Hydra's ``~``
   fails on null-valued keys.

2. **Source encoder is off by default.**  ``module.source_encoder`` is
   ``null``. Only quadratic couplings (``gromov-wasserstein``, ``fused``)
   turn it on via ``_enable_source_encoder``.

3. **Every schema needs a real group column.**  ``groups=[]`` makes the
   loader emit a synthetic ``_sckitflow_all_cells`` axis whose 1-tuple
   weight keys don't match the loader's 2-tuple ``group_by``. Every pair
   therefore defaults ``groups=["source_split"]``.

4. **Coupling costs need a cell-level linear representation.**  The base
   fixture's ``X_repr`` is group-level; the fixture therefore adds
   ``X_coupling_lin`` with i.i.d. noise per cell, and every coupling pair
   uses it as ``rep_lin``.

Coupling → DM reps → VF mapping
-------------------------------
======================  ===========================  ==========================
Coupling                DM reps                      VF source encoder
======================  ===========================  ==========================
``independent``         none                         OFF (either mode)
``exact``               ``rep_lin``                  OFF (linear)
``sinkhorn``            ``rep_lin``                  OFF (linear)
``partial``             ``rep_lin``                  OFF (linear)
``unbalanced``          ``rep_lin``                  OFF (linear)
``gromov-wasserstein``  ``rep_lin`` + both quads     **ON**
``fused``               ``rep_lin`` + both quads     **ON**
======================  ===========================  ==========================

Notes
-----
* ``rep_lin=X_coupling_lin`` (not ``X_repr`` / ``X_src``): the fixture's
  ``X_coupling_lin`` is generated fresh with i.i.d. noise per cell, so the
  coupling cost matrix is non-degenerate regardless of how the base
  fixture shapes its other reps.
* ``gromov-wasserstein`` passes ``rep_lin`` (not ``use_linear=false``):
  ``ot_quadratic_coupling`` unconditionally calls
  ``to_torch_tensor(source_lin)`` and fails on ``None``.
* Runs cap at ``fast_dev_run``; deselect with ``-m "not slow"`` locally.
"""

import math
from pathlib import Path

import anndata as ad
import numpy as np
import pytest
from anndata import AnnData
from hydra import compose, initialize_config_dir

from flowrun import fit_cfm

CONFIG_DIR = Path(fit_cfm.__file__).parent / "configs"

fit_cfm_main = fit_cfm.main


# --------------------------- override rendering ----------------------------


def _render(value) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, str):
        if value in ("", "null", "true", "false") or any(c in value for c in ":[]{},.'\" \t"):
            return "'" + value.replace("'", "''") + "'"
        return value
    if isinstance(value, dict):
        return "{" + ", ".join(f"{_render(k)}: {_render(v)}" for k, v in value.items()) + "}"
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_render(v) for v in value) + "]"
    raise TypeError(f"cannot render {type(value).__name__} as Hydra override")


def overrides(section: str, **fields) -> list[str]:
    """Hydra overrides for **non-null** dict/list fields.

    Uses ``~path`` then ``+path=value`` because Hydra merges dicts; the
    ``~`` half resets the pre-populated value first. ``~path`` fails on
    ``null``-valued keys — for those, use ``set_field``.
    """
    out: list[str] = []
    for key, value in fields.items():
        path = f"{section}.{key}"
        if isinstance(value, (dict, list, tuple)):
            out.append(f"~{path}")
            out.append(f"+{path}={_render(value)}")
        else:
            out.append(f"{path}={_render(value)}")
    return out


def set_field(section: str, key: str, value) -> str:
    """Bare ``path=value`` override.

    Works for ``null``-valued keys — ``overrides()`` fails at compose time
    on those (``~path`` raises "Could not delete from config"). Use this
    for any field whose YAML default is ``null``:
    ``module.condition_encoder``, ``module.source_encoder``, etc.
    """
    return f"{section}.{key}={_render(value)}"


# --------------------------- fixed splitter axis ---------------------------


_OVERRIDES_SPLITTER: list[tuple[str, list[str]]] = [
    (
        "base",
        overrides(
            "splitter",
            group_keys=["source_split"],
            always_train_keys=[],
            source_key="drugA",
            source_value="control",
        ),
    ),
]


# --------------------------- datamanager builder ---------------------------


def _dm_shared(**overrides_kwargs) -> dict:
    """Base datamanager kwargs; each pair extends specific fields."""
    base = {
        "categorical_conditions": {},
        "categorical_reps": {},
        "continuous_conditions": [],
        "control_values_dict": {},
        "groups": [],
        "groups_encoding": {},
        "groups_reps": {},
        "rep_lin": None,
        "source_rep_quad": None,
        "target_rep_quad": None,
        "use_linear": True,
        "target_continuous_covs": [],
    }
    base.update(overrides_kwargs)
    return base


# --------------------------- VF per-condition layers -----------------------


def _per_condition_layers(categorical: list[str], continuous: list[str] = ()) -> str:
    """``module.condition_encoder=...`` override: a set encoder with one input layer per key.

    ``categorical`` are condition and group names, ``continuous`` covariate names from the DM
    config; the input widths come from ``dims``.

    Uses bare ``path=value`` (no ``~``): the YAML default is ``null`` and
    Hydra's ``~`` fails on null-valued keys.
    """
    spec = {
        "output_dim": 8,
        "pooling_proj_dim": 8,
        "categorical_input_layers": {key: {"output_dim": 2, "mlp": {}} for key in categorical},
        "continuous_input_layers": {key: {"output_dim": 2, "mlp": {}} for key in continuous},
    }
    return set_field("module", "condition_encoder", spec)


# --------------------------- override lookups ------------------------------


_OVERRIDES_MODULE_SCALE: list[tuple[str, list[str]]] = [
    (
        "minimal",
        [
            "module.time_features.n_time_features=8",
            "module.state_encoder.output_dim=16",
            "module.time_encoder.output_dim=16",
        ],
    ),
    (
        "medium",
        [
            "module.time_features.n_time_features=16",
            "module.state_encoder.output_dim=32",
            "module.time_encoder.output_dim=32",
        ],
    ),
]


_OVERRIDES_INFERENCE: list[tuple[str, list[str]]] = [
    (
        "fast",
        overrides(
            "inference",
            n_samples=16,
            n_steps=5,
            solver_kwargs={"method": "euler"},
        ),
    ),
    # ``midpoint`` is fixed-step; adaptive solvers hard-code float64
    # for tolerances and raise on MPS.
    (
        "mid",
        overrides(
            "inference",
            n_samples=32,
            n_steps=10,
            solver_kwargs={"method": "midpoint"},
        ),
    ),
]


_OVERRIDES_DMOD: list[tuple[str, list[str]]] = [
    (
        "fast",
        overrides(
            "datamodule",
            batch_size=32,
            n_train_steps=100,
            loader_kwargs={},
        ),
    ),
]


def _lookup(variants, name):
    for variant_name, variant_lines in variants:
        if variant_name == name:
            return variant_lines
    raise KeyError(f"unknown variant {name!r}; available: {[n for n, _ in variants]}")


# The fixture's source_split values, as an explicit label vocabulary.
SOURCE_SPLIT_LABELS = {"type": "group_encoder.label", "classes": [f"source_split{i}" for i in range(5)]}


# --------------------------- schema pairs ----------------------------------
_SCHEMA_PAIRS: dict[str, tuple[list[str], list[str]]] = {
    # ---- Group-only, unpaired -----------------------------------------
    # No categorical_conditions → only the group's input layer.
    "groups_only": (
        overrides(
            "datamanager",
            target_sample_rep="X_repr",
            **_dm_shared(
                groups=["source_split"],
                groups_encoding={"source_split": SOURCE_SPLIT_LABELS},
            ),
        ),
        [_per_condition_layers(["source_split"])],
    ),
    # ---- Continuous-only condition, unpaired --------------------------
    # One covariate key (X_tgt) → one input-layer entry.
    "one_cont_unpaired": (
        overrides(
            "datamanager",
            target_sample_rep="X_repr",
            **_dm_shared(
                continuous_conditions=["X_tgt"],
                groups=["source_split"],
                groups_encoding={"source_split": SOURCE_SPLIT_LABELS},
            ),
        ),
        [_per_condition_layers(["source_split"], ["X_tgt"])],
    ),
    # ---- 1 categorical, unpaired --------------------------------------
    "one_cat_unpaired": (
        overrides(
            "datamanager",
            target_sample_rep="X_repr",
            **_dm_shared(
                categorical_conditions={"drug": ["drugA"]},
                categorical_reps={"drug": "drug"},
            ),
        ),
        [_per_condition_layers(["drug"])],
    ),
    # ---- 1 categorical, paired ----------------------------------------
    "one_cat_paired": (
        overrides(
            "datamanager",
            target_sample_rep="X_repr",
            **_dm_shared(
                categorical_conditions={"drug": ["drugA"]},
                categorical_reps={"drug": "drug"},
                control_values_dict={"drug": "control"},
            ),
        ),
        [_per_condition_layers(["drug"])],
    ),
    "one_cat_paired_with_groups": (
        overrides(
            "datamanager",
            target_sample_rep="X_repr",
            **_dm_shared(
                categorical_conditions={"drug": ["drugA"]},
                categorical_reps={"drug": "drug"},
                control_values_dict={"drug": "control"},
                groups=["source_split"],
                groups_encoding={"source_split": SOURCE_SPLIT_LABELS},
            ),
        ),
        # groups_reps={} means no group dims → no group input-layer entry.
        [_per_condition_layers(["drug", "source_split"])],
    ),
    # ---- 2 categorical, unpaired / paired -----------------------------
    "two_cat_unpaired": (
        overrides(
            "datamanager",
            target_sample_rep="X_repr",
            **_dm_shared(
                categorical_conditions={"drug": ["drugA"], "ko": ["koA"]},
                categorical_reps={"drug": "drug", "ko": "ko"},
            ),
        ),
        [_per_condition_layers(["drug", "ko"])],
    ),
    "two_cat_paired": (
        overrides(
            "datamanager",
            target_sample_rep="X_repr",
            **_dm_shared(
                categorical_conditions={"drug": ["drugA"], "ko": ["koA"]},
                categorical_reps={"drug": "drug", "ko": "ko"},
                control_values_dict={"drug": "control", "ko": "control"},
            ),
        ),
        [_per_condition_layers(["drug", "ko"])],
    ),
    "two_cat_paired_with_groups": (
        overrides(
            "datamanager",
            target_sample_rep="X_repr",
            **_dm_shared(
                categorical_conditions={"drug": ["drugA"], "ko": ["koA"]},
                categorical_reps={"drug": "drug", "ko": "ko"},
                control_values_dict={"drug": "control", "ko": "control"},
                groups=["source_split"],
                groups_encoding={"source_split": SOURCE_SPLIT_LABELS},
            ),
        ),
        [_per_condition_layers(["drug", "ko", "source_split"])],
    ),
    # ---- 1 categorical + 1 continuous, paired -------------------------
    "one_cat_one_cont_paired": (
        overrides(
            "datamanager",
            target_sample_rep="X_repr",
            **_dm_shared(
                categorical_conditions={"drug": ["drugA"]},
                categorical_reps={"drug": "drug"},
                continuous_conditions=["X_tgt"],
                control_values_dict={"drug": "control"},
            ),
        ),
        [_per_condition_layers(["drug"], ["X_tgt"])],
    ),
    "two_cat_one_cont_paired": (
        overrides(
            "datamanager",
            target_sample_rep="X_repr",
            **_dm_shared(
                categorical_conditions={"drug": ["drugA"], "ko": ["koA"]},
                categorical_reps={"drug": "drug", "ko": "ko"},
                continuous_conditions=["X_tgt"],
                control_values_dict={"drug": "control", "ko": "control"},
            ),
        ),
        [_per_condition_layers(["drug", "ko"], ["X_tgt"])],
    ),
    # ---- alternative state representation -----------------------------
    "sample_rep_x_src": (
        overrides(
            "datamanager",
            target_sample_rep="X_src",
            **_dm_shared(
                categorical_conditions={"drug": ["drugA"]},
                categorical_reps={"drug": "drug"},
                control_values_dict={"drug": "control"},
            ),
        ),
        [_per_condition_layers(["drug"])],
    ),
}


# --------------------------- coupling pairs --------------------------------


def _enable_source_encoder() -> list[str]:
    """Enable the source encoder for quadratic couplings.

    A bare ``key=value`` override (no ``~``) because the YAML sets
    ``module.source_encoder`` to ``null``; Hydra's ``~`` (delete) fails on
    null-valued keys.
    """
    return ["module.source_encoder={output_dim: 16, mlp: {hidden_dims: [16], activation_cls: silu}}"]


def _coupling_pair(
    method: str,
    *,
    base_schema: str = "one_cat_paired",
    dm_extra: dict | None = None,
    vf_overrides: list[str] | None = None,
) -> list[str]:
    """Compose overrides for a coupling + base schema pair.

    Order matters for Hydra's last-override-wins:
        splitter → dm → fast dmod → module_scale → schema overrides
        → coupling.method → dm_extra → vf_overrides
    """
    dm_over, module_schema_over = _SCHEMA_PAIRS[base_schema]
    out = (
        _OVERRIDES_SPLITTER[0][1]
        + dm_over
        + _lookup(_OVERRIDES_DMOD, "fast")
        + _lookup(_OVERRIDES_MODULE_SCALE, "minimal")
        + module_schema_over
        + [f"coupling.method={method}"]
    )
    if dm_extra:
        out += overrides("datamanager", **dm_extra)
    if vf_overrides:
        out += vf_overrides
    return out


_COUPLING_PAIRS: dict[str, list[str]] = {
    # ---- Independent — no reps needed, source encoder off -------------
    "independent_unpaired": _coupling_pair(
        "independent",
        base_schema="one_cat_unpaired",
    ),
    "independent_paired": _coupling_pair(
        "independent",
        base_schema="one_cat_paired",
    ),
    # ---- Linear OT couplings — rep_lin, source encoder OFF -----------
    "exact": _coupling_pair(
        "exact",
        base_schema="one_cat_paired",
        dm_extra={"rep_lin": "X_coupling_lin"},
    ),
    "sinkhorn": _coupling_pair(
        "sinkhorn",
        base_schema="one_cat_paired",
        dm_extra={"rep_lin": "X_coupling_lin"},
    ),
    "partial": _coupling_pair(
        "partial",
        base_schema="one_cat_paired",
        dm_extra={"rep_lin": "X_coupling_lin"},
    ),
    "unbalanced": _coupling_pair(
        "unbalanced",
        base_schema="one_cat_paired",
        dm_extra={"rep_lin": "X_coupling_lin"},
    ),
    # ---- Gromov-Wasserstein — linear + quads, source encoder ON ------
    "gromov_wasserstein": _coupling_pair(
        "gromov-wasserstein",
        base_schema="one_cat_paired",
        dm_extra={
            "rep_lin": "X_coupling_lin",
            "source_rep_quad": "X_src_quad",
            "target_rep_quad": "X_tgt_quad",
        },
        vf_overrides=_enable_source_encoder(),
    ),
    # ---- Fused — linear + quads blended by `alpha` -------------------
    "fused": _coupling_pair(
        "fused",
        base_schema="one_cat_paired",
        dm_extra={
            "rep_lin": "X_coupling_lin",
            "source_rep_quad": "X_src_quad",
            "target_rep_quad": "X_tgt_quad",
        },
        vf_overrides=_enable_source_encoder(),
    ),
}


# --------------------------- orthogonal axes --------------------------------


def combine(
    *,
    pair: str = "one_cat_paired",
    splitter: str = "base",
    dmod: str = "fast",
    module_scale: str = "minimal",
    inference: str = "fast",
    **extras: list[str],
) -> list[str]:
    """Assemble Hydra overrides for a single (DM, VF) schema pair."""
    dm_over, module_schema_over = _SCHEMA_PAIRS[pair]
    out = (
        _lookup(_OVERRIDES_SPLITTER, splitter)
        + dm_over
        + _lookup(_OVERRIDES_DMOD, dmod)
        + _lookup(_OVERRIDES_MODULE_SCALE, module_scale)
        + module_schema_over
        + _lookup(_OVERRIDES_INFERENCE, inference)
    )
    for extra in extras.values():
        out += extra
    return out


# --------------------------- composition -----------------------------------


def _compose(*, target_adata_path, tmp_path, overrides, config_name="fit_cfm"):
    ephemeral = [
        f"paths.target_adata={target_adata_path}",
        f"save.run_dir={tmp_path / 'run'}",
        f"callbacks.model_checkpoint.dirpath={tmp_path / 'ckpt'}",
        f"loggers.wandb.save_dir={tmp_path / 'wandb'}",
        f"loggers.csv.save_dir={tmp_path / 'csv'}",
        f"trainer.default_root_dir={tmp_path / 'logs'}",
        "trainer.fast_dev_run=true",
        "trainer.enable_progress_bar=false",
        # the dummy data's few combinations leave val empty at the default fraction; test is always held out
        "callbacks.model_checkpoint.monitor=test/edist",
        "loggers.wandb.offline=true",
        "run.allow_overwrite=true",
        "hydra.job.chdir=false",
    ]
    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
        return compose(config_name=config_name, overrides=overrides + ephemeral)


# --------------------------- fixtures --------------------------------------


@pytest.fixture
def target_adata_path(tmp_path: Path, adata: AnnData) -> Path:
    adata.obs_names = adata.obs_names.astype(str)
    adata.var_names = adata.var_names.astype(str)
    rng = np.random.default_rng(42)
    # Cell-level linear rep for the coupling cost matrix.
    adata.obsm["X_coupling_lin"] = rng.standard_normal((adata.n_obs, 8)).astype("float32")
    # Quad reps for quadratic couplings.
    adata.obsm["X_src_quad"] = rng.standard_normal((adata.n_obs, 4)).astype("float32")
    adata.obsm["X_tgt_quad"] = rng.standard_normal((adata.n_obs, 6)).astype("float32")
    path = tmp_path / "target.h5ad"
    adata.write_h5ad(path)
    return path


# --------------------------- shared assertions -----------------------------


def _run_and_assert(
    *,
    overrides_list: list[str],
    tmp_path: Path,
    target_adata_path: Path,
) -> None:
    cfg = _compose(
        target_adata_path=target_adata_path,
        tmp_path=tmp_path,
        overrides=overrides_list,
    )

    try:
        score = fit_cfm_main.__wrapped__(cfg)
    except Exception:
        fit_cfm.logger.exception("fit_cfm.main() raised")
        raise

    assert isinstance(score, float)
    assert not math.isnan(score)

    run_dir = tmp_path / "run"
    assert (run_dir / "config.json").is_file()


# --------------------------- tests -----------------------------------------


class TestScriptLoads:
    def test_script_loads(self) -> None:
        assert hasattr(fit_cfm, "main")
        assert callable(fit_cfm_main.__wrapped__)
        assert fit_cfm.logger.name == "flowrun.fit_cfm"


class TestFitCFMRunsSchemaPairs:
    """Schema matrix: categorical_conditions × covariates × paired/unpaired, all with groups.

    Every pair supplies ``module.condition_encoder`` with one input layer per
    condition key its DM produces, so the per-condition input-layer path is
    exercised in ``NeuralVelocityConfig.build`` and ``NeuralVelocity`` end-to-end.
    """

    @pytest.mark.slow
    @pytest.mark.parametrize("pair", list(_SCHEMA_PAIRS), ids=list(_SCHEMA_PAIRS))
    def test_schema_pair(
        self,
        pair: str,
        tmp_path: Path,
        target_adata_path: Path,
    ) -> None:
        _run_and_assert(
            overrides_list=combine(pair=pair),
            tmp_path=tmp_path,
            target_adata_path=target_adata_path,
        )


class TestFitCFMRunsCouplings:
    """Coupling matrix: each method paired with the DM reps + VF it needs."""

    @pytest.mark.slow
    @pytest.mark.parametrize("coupling_name", list(_COUPLING_PAIRS), ids=list(_COUPLING_PAIRS))
    def test_coupling(
        self,
        coupling_name: str,
        tmp_path: Path,
        target_adata_path: Path,
    ) -> None:
        _run_and_assert(
            overrides_list=_COUPLING_PAIRS[coupling_name],
            tmp_path=tmp_path,
            target_adata_path=target_adata_path,
        )


class TestFitCFMRunsAxes:
    """Orthogonal axes on top of ``pair="one_cat_paired"``."""

    @pytest.mark.slow
    @pytest.mark.parametrize(
        "overrides_list",
        [
            pytest.param(
                combine(pair="one_cat_paired", module_scale="minimal"),
                id="scale-minimal",
            ),
            pytest.param(
                combine(pair="one_cat_paired", module_scale="medium"),
                id="scale-medium",
            ),
            pytest.param(
                combine(pair="one_cat_paired", inference="fast"),
                id="inference-fast",
            ),
            pytest.param(
                combine(pair="one_cat_paired", inference="mid"),
                id="inference-mid",
            ),
            pytest.param(
                combine(
                    pair="one_cat_paired",
                    datamanager=[
                        "datamanager.use_linear=false",
                        "datamanager.source_rep_quad=X_src_quad",
                        "datamanager.target_rep_quad=X_tgt_quad",
                    ],
                ),
                id="use-linear-false",
            ),
            pytest.param(
                combine(
                    pair="one_cat_paired",
                    probability_path=["probability_path.name=linear_dirac"],
                ),
                id="linear-dirac",
            ),
            pytest.param(
                combine(
                    pair="one_cat_paired",
                    probability_path=["probability_path.name=schrodinger_bridge"],
                ),
                id="schrodinger-bridge",
            ),
        ],
    )
    def test_axis(
        self,
        overrides_list: list[str],
        tmp_path: Path,
        target_adata_path: Path,
    ) -> None:
        _run_and_assert(
            overrides_list=overrides_list,
            tmp_path=tmp_path,
            target_adata_path=target_adata_path,
        )


def test_holdout_split_with_preprocessing_from_zarr(tmp_path: Path, target_adata_path: Path) -> None:
    """Read a zarr input, hold out whole values of one column, and train on a PCA fitted on the training split."""
    zarr_path = tmp_path / "target.zarr"
    ad.read_h5ad(target_adata_path).write_zarr(zarr_path)
    _run_and_assert(
        overrides_list=[
            *combine(pair="one_cat_paired"),
            "splitter.type=shared_source_holdout",
            "splitter.key=source_split",
            "splitter.test=[source_split0]",
            "splitter.val=[source_split1]",
            "preprocessing.pca=5",
            "preprocessing.zscore=true",
            "preprocessing.whitening=zca",
            "preprocessing.overwrite=true",  # the PCA replaces the schema's existing target_sample_rep
        ],
        tmp_path=tmp_path,
        target_adata_path=zarr_path,
    )
    written = ad.read_zarr(zarr_path)  # every cell projected into the schema's target_sample_rep, basis alongside
    assert written.obsm["X_repr"].shape == (written.n_obs, 5)
    assert written.uns["preprocessing"]["pca"]["PCs"].shape == (written.n_vars, 5)
    assert list(written.uns["preprocessing"]["steps"]) == ["pca", "zscore", "whitening"]
