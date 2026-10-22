"""The panel imputation experiments (ported from theislab/sckitflow#202)."""

import json
import math
import re
import subprocess
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import yaml
from hydra import compose, initialize_config_dir

from flowrun import fit_cfm, panels, split

CONFIGS = str(Path(fit_cfm.__file__).parent / "configs")


@pytest.fixture
def tiny_panels(tmp_path: Path) -> list[Path]:
    """Two prepared panels: named markers (6 shared after renaming), a label and its one-hot table, `logicle_trans`."""
    rng = np.random.default_rng(0)
    shared = [f"shared_{i}" for i in range(6)]
    var = {"old": [*shared, "CD201", "old_0", "old_1"], "new": [*shared, "CD201", "new_0", "new_1", "new_2"]}
    onehot = {"new panel": np.array([1.0, 0.0]), "old panel": np.array([0.0, 1.0])}
    paths = []
    for name, label in [("old", "old panel"), ("new", "new panel")]:
        logicle = rng.normal(size=(200, len(var[name]))).astype(np.float32)
        obs = pd.DataFrame({"dataset": pd.Categorical([label] * 200, categories=sorted(onehot))})
        obs.index = [f"{name}_{i}" for i in range(200)]
        a = ad.AnnData(X=logicle * 2, obs=obs, var=pd.DataFrame(index=var[name]), layers={"logicle_trans": logicle})
        a.uns["dataset"] = onehot
        paths.append(tmp_path / "prepared" / f"{name}.zarr")
        a.write_zarr(paths[-1])
    return paths


def test_two_runs_in_one_process_get_their_own_wandb_runs(tiny_panels: list[Path], tmp_path: Path) -> None:
    """Each fit finishes its W&B run; otherwise the next fit in the process (in-process sweeps) logs into it."""
    for name in ("a", "b"):
        overrides = [
            "+experiment=flow_cyto_panel_alignment_smoke",
            f"paths.target_adata=[{tiny_panels[0]},{tiny_panels[1]}]",
            f"save.run_dir={tmp_path / name / 'run'}",
            f"callbacks.model_checkpoint.dirpath={tmp_path / name / 'ckpt'}",
            f"loggers.wandb.save_dir={tmp_path / name / 'wandb'}",
            f"loggers.csv.save_dir={tmp_path / name / 'csv'}",
            f"trainer.default_root_dir={tmp_path / name / 'logs'}",
            "hydra.job.chdir=false",
        ]
        with initialize_config_dir(config_dir=CONFIGS, version_base=None):
            fit_cfm.main.__wrapped__(compose(config_name="fit_cfm", overrides=overrides))
    for name in ("a", "b"):
        assert len(list((tmp_path / name / "wandb").rglob("config_resolved.yaml"))) == 1


@pytest.fixture
def prepared_by_script(tmp_path: Path) -> list[Path]:
    """Two original panels (h5ad) with a synonym pair, prepared to zarr by `python -m flowrun.panels`."""
    rng = np.random.default_rng(0)
    shared = [f"shared_{i}" for i in range(6)]
    var = {"old": [*shared, "EPCR", "old_0", "old_1"], "new": [*shared, "CD201", "new_0", "new_1", "new_2"]}
    args = ["--out", str(tmp_path / "prepared"), "--rename", "EPCR=CD201"]
    for name, label in [("old", "old panel"), ("new", "new panel")]:
        logicle = rng.normal(size=(200, len(var[name]))).astype(np.float32)
        a = ad.AnnData(X=logicle * 2, var=pd.DataFrame(index=var[name]), layers={"logicle_trans": logicle})
        a.obs_names = [f"{name}_{i}" for i in range(a.n_obs)]
        a.write_h5ad(tmp_path / f"{name}.h5ad")
        args += ["--panel", label, str(tmp_path / f"{name}.h5ad")]
    panels.main(args)
    return [tmp_path / "prepared" / "old.zarr", tmp_path / "prepared" / "new.zarr"]


def test_prepare_panels_renames_labels_and_records(prepared_by_script: list[Path]) -> None:
    old, new = (ad.read_zarr(p) for p in prepared_by_script)
    assert "CD201" in old.var_names and "EPCR" not in old.var_names
    assert old.var.loc["CD201", "original_name"] == "EPCR"
    assert set(old.obs["dataset"]) == {"old panel"} and set(new.obs["dataset"]) == {"new panel"}
    np.testing.assert_array_equal(new.uns["dataset"]["new panel"], [1, 0])  # sorted labels, as in #202
    assert dict(old.uns["panels"]["rename"]) == {"EPCR": "CD201"}
    assert not set(old.obsm) & {"state_from_layer", "shared_features", "own_features"}  # derived at load time


def test_coupling_terms_from_the_state_by_feature_name(tiny_panels: list[Path]) -> None:
    old, new = (ad.read_zarr(p) for p in tiny_panels)
    state = {"target_sample_rep": "layers/logicle_trans", "source_sample_rep": "layers/logicle_trans"}
    dm, features = panels.from_state([old, new], {**state, "lin_from_state": True, "quad_from_state": True})
    assert features["shared"] == sorted([*(f"shared_{i}" for i in range(6)), "CD201"])
    assert features["own"] == [["old_0", "old_1"], ["new_0", "new_1", "new_2"]]
    assert dm["target_sample_rep"] == dm["source_sample_rep"] and dm["rep_lin"] and dm["source_rep_quad"]
    assert "lin_from_state" not in dm and "quad_from_state" not in dm  # sckitflow gets plain reps
    col = list(new.var_names)
    np.testing.assert_array_equal(
        new.obsm[dm["rep_lin"]][:, features["shared"].index("CD201")],
        new.layers["logicle_trans"][:, col.index("CD201")],
    )
    np.testing.assert_array_equal(new.obsm[dm["target_sample_rep"]], new.layers["logicle_trans"])
    assert old.obsm[dm["source_rep_quad"]].shape == (200, 2) and new.obsm[dm["target_rep_quad"]].shape == (200, 3)

    lin_only, _ = panels.from_state([old, new], {**state, "lin_from_state": True})
    assert lin_only.get("source_rep_quad") is None


@pytest.mark.parametrize(
    ("dm", "match"),
    [
        ({"target_sample_rep": "X_pca", "lin_from_state": True}, "the state must be `X` or `layers/<key>`"),
        ({"target_sample_rep": "X", "rep_lin": "X_pca", "lin_from_state": True}, "either rep_lin or lin_from_state"),
        ({"target_sample_rep": "X", "target_rep_quad": "X_q", "quad_from_state": True}, "or quad_from_state, not both"),
    ],
)
def test_coupling_terms_from_the_state_refuse(tiny_panels: list[Path], dm: dict, match: str) -> None:
    with pytest.raises(ValueError, match=re.escape(match)):
        panels.from_state([ad.read_zarr(p) for p in tiny_panels], dm)


def test_random_cells_splits_targets_per_group_and_shares_sources() -> None:
    obs = pd.DataFrame({"dataset": ["old panel"] * 50 + ["new panel"] * 100})
    labels = split.random_cells(
        obs, ["dataset"], is_source=(obs["dataset"] == "old panel").to_numpy(), test_fraction=0.2, val_fraction=0.1
    )
    assert labels[:50].eq("all").all()
    assert labels[50:].astype(str).value_counts().to_dict() == {"train": 70, "test": 20, "val": 10}


def test_panel_smoke_runs(tiny_panels: list[Path], tmp_path: Path) -> None:
    overrides = [
        "+experiment=flow_cyto_panel_alignment_smoke",
        f"paths.target_adata=[{tiny_panels[0]},{tiny_panels[1]}]",
        f"save.run_dir={tmp_path / 'run'}",
        f"callbacks.model_checkpoint.dirpath={tmp_path / 'ckpt'}",
        f"loggers.wandb.save_dir={tmp_path / 'wandb'}",
        f"loggers.csv.save_dir={tmp_path / 'csv'}",
        f"trainer.default_root_dir={tmp_path / 'logs'}",
        "hydra.job.chdir=false",
    ]
    with initialize_config_dir(config_dir=CONFIGS, version_base=None):
        cfg = compose(config_name="fit_cfm", overrides=overrides)
    assert cfg.coupling.method == "fused" and cfg.module.conditioning_id == "resnet1d"
    score = fit_cfm.main.__wrapped__(cfg)
    assert math.isfinite(score)  # the monitored val/edist was logged, not a missing metric
    assert (tmp_path / "run" / "config.json").is_file()
    old, new = (ad.read_zarr(p) for p in tiny_panels)
    assert set(old.obs["split_42"]) == {"all"}  # the seed's split, written back; sources are every split's
    assert new.obs["split_42"].astype(str).value_counts().to_dict() == {"train": 140, "test": 40, "val": 20}
    assert not set(new.obsm) & {"state_from_layer", "shared_features", "own_features"}  # the reps stay in memory
    hparams = yaml.safe_load(next((tmp_path / "csv").rglob("hparams.yaml")).read_text())
    assert hparams["coupling"]["method"] == "fused" and hparams["datamanager"]["lin_from_state"]  # resolved config
    summary = json.loads((tmp_path / "run" / "summary.json").read_text())
    assert summary["params/coupling.method"] == "fused" and summary["params/run.splitter_seed"] == 42
    assert {"best/val/edist", "best/test/edist", "best/epoch"} <= set(summary)  # the best epoch's, for sorting
    assert (tmp_path / "run" / "config_resolved.yaml").is_file()
    (resolved,) = (tmp_path / "wandb").rglob("config_resolved.yaml")
    assert yaml.safe_load(resolved.read_text())["splitter"]["split_key"] == "split_42"  # interpolations resolved


def test_a_never_logged_metric_fails_the_run(tiny_panels: list[Path], tmp_path: Path) -> None:
    """Raised, so a sweep counts the trial as failed; a returned nan crashes the Optuna sweeper instead."""
    overrides = [
        "+experiment=flow_cyto_panel_alignment_smoke",
        f"paths.target_adata=[{tiny_panels[0]},{tiny_panels[1]}]",
        f"save.run_dir={tmp_path / 'run'}",
        f"loggers.wandb.save_dir={tmp_path / 'wandb'}",
        f"loggers.csv.save_dir={tmp_path / 'csv'}",
        f"trainer.default_root_dir={tmp_path / 'logs'}",
        f"callbacks.model_checkpoint.dirpath={tmp_path / 'ckpt'}",
        "callbacks.model_checkpoint.monitor=val/missing",
        "hydra.job.chdir=false",
    ]
    with initialize_config_dir(config_dir=CONFIGS, version_base=None):
        cfg = compose(config_name="fit_cfm", overrides=overrides)
    with pytest.raises(RuntimeError, match=r"'val/missing' was never logged"):
        fit_cfm.main.__wrapped__(cfg)


def test_the_panel_contract() -> None:
    """The panel experiment as ported from theislab/sckitflow#202."""
    with initialize_config_dir(config_dir=CONFIGS, version_base=None):
        cfg = compose(config_name="fit_cfm", overrides=["+experiment=flow_cyto_panel_alignment"])
    c = cfg.coupling
    assert (c.method, c.epsilon, c.alpha, c.scale_cost, c.max_iter, c.tol, c.inner_stop_thr) == (
        "fused",
        1.0,
        0.5,
        "median",
        100,
        1e-5,
        1e-4,
    )
    dm = cfg.datamanager
    assert dm.target_sample_rep == dm.source_sample_rep == "layers/logicle_trans"
    assert dm.lin_from_state and dm.quad_from_state and cfg.splitter.type == "random_cells"
    assert dm.split_by == cfg.splitter.split_key == "split_42"
    assert dict(dm.control_values_dict) == {"panel": "old panel"}
    assert dict(dm.categorical_reps) == {"panel": "dataset"}
    assert cfg.module.conditioning_id == "resnet1d" and cfg.module.source_encoder_output_dim == 16
    assert cfg.training.generate_from_noise and cfg.inference.generate_from_noise
    assert cfg.callbacks.model_checkpoint.monitor == cfg.callbacks.early_stopping.monitor == "val/edist"


FUSED = {
    "coupling.epsilon": "choice(0.5, 1.0, 2.0)",
    "coupling.alpha": "choice(0.25, 0.5, 0.75)",
    "coupling.scale_cost": "choice(mean, median)",
}
LINEAR = {"coupling.reg": "choice(0.05, 0.1, 0.5)", "coupling.scale_cost": "choice(mean, median)"}


@pytest.mark.parametrize(
    ("experiment", "axes"),
    [
        ("flow_cyto_panel_alignment_sweep_fused", FUSED),
        ("flow_cyto_panel_alignment_sweep_linear", LINEAR),
        ("flow_cyto_panel_alignment_sweep_smoke", FUSED),
    ],
)
def test_the_sweep_composes(experiment: str, axes: dict, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("FLOWRUN_SWEEP_DIR", raising=False)
    with initialize_config_dir(config_dir=CONFIGS, version_base=None):
        cfg = compose(config_name="fit_cfm", overrides=[f"+experiment={experiment}"], return_hydra_config=True)
    sweeper, launcher = cfg.hydra.sweeper, cfg.hydra.launcher
    assert sweeper._target_ == "hydra_plugins.hydra_optuna_sweeper.optuna_sweeper.OptunaSweeper"
    assert launcher._target_ == "hydra._internal.core_plugins.basic_launcher.BasicLauncher"  # a project picks its own
    assert sweeper.sampler._target_ == "optuna.samplers.RandomSampler"
    assert sweeper.sampler.seed == 0
    assert sweeper.direction == "minimize" and sweeper.max_failure_rate == 0.2
    assert sweeper.n_trials == sweeper.n_jobs == (1 if experiment.endswith("smoke") else 30)
    assert dict(sweeper.params) == {
        "run.splitter_seed": "choice(42, 43, 44, 45, 46)",
        **axes,
    }
    assert cfg.coupling.method == ("sinkhorn" if experiment.endswith("linear") else "fused")
    if experiment.endswith("linear"):  # matched on the shared markers only
        assert not cfg.datamanager.quad_from_state
    assert cfg.loggers.wandb.init_kwargs.group == experiment
    # a fresh timestamped directory per launch, so a relaunch never collides with finished trials
    assert re.fullmatch(rf"multirun/{experiment}/\d{{4}}-\d\d-\d\d_\d\d-\d\d-\d\d", cfg.hydra.sweep.dir)


def test_console_script_finds_its_configs(tmp_path: Path) -> None:
    """The installed entry point, not `python -m`: Hydra then looks the configs up as the `flowrun.configs` package."""
    script = Path(sys.executable).parent / "flowrun-fit-cfm"
    result = subprocess.run(
        [str(script), "--cfg", "all", "+experiment=flow_cyto_panel_alignment_sweep_smoke"],
        check=False,
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, result.stderr[-3000:]
    assert "method: fused" in result.stdout and "OptunaSweeper" in result.stdout


@pytest.mark.slow
def test_one_sweep_trial_runs_through_optuna(tiny_panels: list[Path], tmp_path: Path) -> None:
    """The real sweeper and sampler, with Hydra's local launcher standing in for SLURM and the tiny data."""
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "flowrun.fit_cfm",
            "-m",
            "+experiment=flow_cyto_panel_alignment_sweep_smoke",
            "hydra/launcher=basic",
            f"hydra.sweep.dir={tmp_path / 'sweep'}",
            f"paths.target_adata=[{tiny_panels[0]},{tiny_panels[1]}]",  # the tiny data instead
            "hydra.sweeper.params={run.splitter_seed:'choice(0)'}",
            "trainer.accelerator=cpu",
            "trainer.fast_dev_run=true",
            "datamodule.batch_size=16",
            "inference.n_samples=2",
        ],
        check=False,
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert result.returncode == 0, result.stderr[-3000:]
    assert "infer_relative_search_space" not in result.stderr
    results = yaml.safe_load((tmp_path / "sweep" / "optimization_results.yaml").read_text())
    assert np.isfinite(results["best_value"])  # Optuna got the monitored val/edist


@pytest.mark.slow
def test_a_grid_runs_as_parallel_local_processes(tiny_panels: list[Path], tmp_path: Path) -> None:
    """Hydra's joblib launcher: one process per grid point, each with its own output directory."""
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "flowrun.fit_cfm",
            "-m",
            "+experiment=flow_cyto_panel_alignment_smoke",
            "hydra/launcher=joblib",
            "hydra.launcher.n_jobs=2",
            f"hydra.sweep.dir={tmp_path / 'grid'}",
            "paths.out_dir=${hydra:runtime.output_dir}",
            f"paths.target_adata=[{tiny_panels[0]},{tiny_panels[1]}]",
            "coupling.epsilon=0.5,1.0",
        ],
        check=False,
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert result.returncode == 0, result.stderr[-3000:]
    assert sorted(p.parent.parent.name for p in (tmp_path / "grid").glob("*/save/config.json")) == ["0", "1"]
