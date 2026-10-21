"""The panel imputation experiments (ported from theislab/sckitflow#202)."""

import subprocess
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import yaml
from hydra import compose, initialize_config_dir

from flowrun import fit_cfm

CONFIGS = str(Path(fit_cfm.__file__).parent / "configs")


@pytest.fixture
def tiny_panel_adata(tmp_path: Path) -> Path:
    """Two panels sharing some markers, with the obsm keys, the panel table and the 80/20 split the experiment reads."""
    rng = np.random.default_rng(0)
    n_old = n_new = 200
    shared, old_only, new_only = (
        [f"shared_{i}" for i in range(10)],
        [f"old_{i}" for i in range(5)],
        [f"new_{i}" for i in range(5)],
    )
    var_names = shared + old_only + new_only
    X = rng.normal(size=(n_old + n_new, len(var_names))).astype(np.float32)
    X[n_old:, len(shared) : len(shared) + len(old_only)] = 0.0  # the new panel does not measure the old markers
    X[:n_old, len(shared) + len(old_only) :] = 0.0
    split = (["train"] * 160 + ["test"] * 40) * 2
    obs = pd.DataFrame(
        {"dataset": ["old panel"] * n_old + ["new panel"] * n_new, "split": split},
        index=[f"cell_{i}" for i in range(n_old + n_new)],
    )
    adata = ad.AnnData(X=X, obs=obs, var=pd.DataFrame(index=var_names))
    adata.obsm["X_zscore"] = rng.normal(size=(adata.n_obs, 12)).astype(np.float32)
    adata.obsm["X_lin"] = rng.normal(size=(adata.n_obs, 12)).astype(np.float32)
    adata.obsm["X_quad_src"] = rng.normal(size=(adata.n_obs, 6)).astype(np.float32)
    adata.obsm["X_quad_tgt"] = rng.normal(size=(adata.n_obs, 6)).astype(np.float32)
    adata.uns["dataset"] = {"old panel": np.array([1.0, 0.0]), "new panel": np.array([0.0, 1.0])}
    path = tmp_path / "adata_concat.zarr"
    adata.write_zarr(path)
    return path


def test_panel_smoke_runs(tiny_panel_adata: Path, tmp_path: Path) -> None:
    overrides = [
        "+experiment=panel_smoke",
        f"paths.target_adata={tiny_panel_adata}",
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
    assert isinstance(score, float)
    assert (tmp_path / "run" / "config.json").is_file()
    assert ad.read_zarr(tiny_panel_adata).obs["split"].value_counts()["test"] == 80  # the prepared split was reused


@pytest.mark.parametrize("experiment", ["panel_sweep", "panel_sweep_smoke"])
def test_the_sweep_composes(experiment: str) -> None:
    with initialize_config_dir(config_dir=CONFIGS, version_base=None):
        cfg = compose(config_name="fit_cfm", overrides=[f"+experiment={experiment}"], return_hydra_config=True)
    sweeper, launcher = cfg.hydra.sweeper, cfg.hydra.launcher
    assert sweeper._target_ == "hydra_plugins.hydra_optuna_sweeper.optuna_sweeper.OptunaSweeper"
    assert launcher._target_ == "hydra_plugins.hydra_submitit_launcher.submitit_launcher.SlurmLauncher"
    assert sweeper.sampler._target_ == "optuna.samplers.RandomSampler"
    assert sweeper.n_trials == (1 if experiment.endswith("smoke") else 30)
    assert set(sweeper.params) == {"paths.target_adata", "coupling.epsilon", "coupling.alpha", "coupling.scale_cost"}


def test_one_sweep_trial_runs_through_optuna(tiny_panel_adata: Path, tmp_path: Path) -> None:
    """The real sweeper and sampler, with Hydra's local launcher standing in for SLURM and the tiny data."""
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "flowrun.fit_cfm",
            "-m",
            "+experiment=panel_sweep_smoke",
            "hydra/launcher=basic",
            f"hydra.sweep.dir={tmp_path / 'sweep'}",
            f"hydra.sweeper.params={{paths.target_adata:'choice({tiny_panel_adata})'}}",  # the tiny data instead
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
    assert np.isfinite(results["best_value"])  # Optuna got the monitored test/edist, not the inf of a missing val/edist
