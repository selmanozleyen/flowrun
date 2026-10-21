"""The panel imputation experiments (ported from theislab/sckitflow#202)."""

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
    """Two original panels (h5ad) with a synonym pair, prepared to zarr by the `flowrun.panels` script."""
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


def test_prepare_panels_renames_labels_and_records(tiny_panels: list[Path]) -> None:
    old, new = (ad.read_zarr(p) for p in tiny_panels)
    assert "CD201" in old.var_names and "EPCR" not in old.var_names
    assert old.var.loc["CD201", "original_name"] == "EPCR"
    assert set(old.obs["dataset"]) == {"old panel"} and set(new.obs["dataset"]) == {"new panel"}
    np.testing.assert_array_equal(new.uns["dataset"]["new panel"], [1, 0])  # sorted labels, as in #202
    assert dict(old.uns["panels"]["rename"]) == {"EPCR": "CD201"}
    assert not set(old.obsm) & {"X_state", "X_lin", "X_quad"}  # the reps follow from the names at load time


def test_panel_reps_by_marker_name(tiny_panels: list[Path]) -> None:
    old, new = (ad.read_zarr(p) for p in tiny_panels)
    reps = panels.panel_reps([old, new], layer="logicle_trans")
    assert reps["shared"] == sorted([*(f"shared_{i}" for i in range(6)), "CD201"])
    assert reps["own"] == [["old_0", "old_1"], ["new_0", "new_1", "new_2"]]
    i = list(new.var_names).index("CD201")
    np.testing.assert_array_equal(
        new.obsm["X_lin"][:, reps["shared"].index("CD201")], new.layers["logicle_trans"][:, i]
    )
    assert old.obsm["X_state"].shape == (200, 9) and new.obsm["X_quad"].shape == (200, 3)


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
    assert not set(new.obsm) & {"X_state", "X_lin", "X_quad"}  # the reps stay in memory


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
    assert (c.method, c.epsilon, c.alpha, c.scale_cost, c.max_iter, c.tol) == ("fused", 1.0, 0.5, "median", 1000, 1e-9)
    dm = cfg.datamanager
    assert (dm.target_sample_rep, dm.source_sample_rep, dm.rep_lin, dm.source_rep_quad, dm.target_rep_quad) == (
        "X_state",
        "X_state",
        "X_lin",
        "X_quad",
        "X_quad",
    )
    assert cfg.panels.layer == "logicle_trans" and cfg.splitter.type == "random_cells"
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
    assert launcher._target_ == "hydra_plugins.hydra_submitit_launcher.submitit_launcher.SlurmLauncher"
    assert sweeper.sampler._target_ == "optuna.samplers.RandomSampler"
    assert sweeper.sampler.seed == 0
    assert sweeper.direction == "minimize" and sweeper.max_failure_rate == 0.2
    assert sweeper.n_trials == sweeper.n_jobs == (1 if experiment.endswith("smoke") else 30)
    assert dict(sweeper.params) == {
        "run.splitter_seed": "choice(42, 43, 44, 45, 46)",
        **axes,
    }
    assert (launcher.partition, launcher.qos, launcher.gres, launcher.timeout_min) == (
        "gpu_p",
        "gpu_normal",
        "gpu:1",
        720,
    )
    assert cfg.coupling.method == ("sinkhorn" if experiment.endswith("linear") else "fused")
    if experiment.endswith("linear"):  # matched on the shared markers only
        assert cfg.datamanager.source_rep_quad is None and cfg.datamanager.target_rep_quad is None
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
    assert "method: fused" in result.stdout and "SlurmLauncher" in result.stdout


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
