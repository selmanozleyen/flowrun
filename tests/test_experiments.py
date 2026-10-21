from pathlib import Path

import anndata as ad
import numpy as np
import pytest
import scipy.sparse as sp
from hydra import compose, initialize_config_dir
from sckitflow.data.sim import get_dummy_adata

from flowrun import fit_cfm


@pytest.mark.parametrize("experiment", ["cellflow_smoke", "[cellflow_smoke,genot]"])
def test_cellflow_smoke_runs_on_cellflow_shaped_zarr(tmp_path: Path, experiment: str):
    """The CellFlow experiment on its expected data layout: sparse X in zarr, drug1 / drug2 / cell_line."""
    adata = get_dummy_adata(
        n_obs_pert=400,
        n_obs_ctrl=100,
        n_genes=60,
        obsm_keys_to_dim={},
        obs_columns_to_nunique_and_prefix={"drug1": (3, "drug"), "drug2": (3, "drug"), "cell_line": (2, "cell_line")},
        uns_keys_to_nunique_prefix_and_dim={"drug": (3, "drug", 8), "cell_line": (2, "cell_line", 4)},
        obs_columns_to_fixed_val={"drug1": "control", "drug2": "control"},
        rng=np.random.default_rng(0),
    )
    adata.X = sp.csr_matrix(np.log1p(np.abs(adata.X)).astype(np.float32))
    adata.obs_names = adata.obs_names.astype(str)
    adata.write_zarr(tmp_path / "data.zarr")

    overrides = [
        f"+experiment={experiment}",
        f"paths.target_adata={tmp_path / 'data.zarr'}",
        f"save.run_dir={tmp_path / 'run'}",
        f"callbacks.model_checkpoint.dirpath={tmp_path / 'ckpt'}",
        f"loggers.wandb.save_dir={tmp_path / 'wandb'}",
        f"loggers.csv.save_dir={tmp_path / 'csv'}",
        f"trainer.default_root_dir={tmp_path / 'logs'}",
        "trainer.fast_dev_run=true",
        "trainer.enable_progress_bar=false",
        "splitter.test_fraction=0.3",
        "splitter.val_fraction=0.3",
        "hydra.job.chdir=false",
    ]

    def run(*extra: str) -> ad.AnnData:
        with initialize_config_dir(config_dir=str(Path(fit_cfm.__file__).parent / "configs"), version_base=None):
            fit_cfm.main.__wrapped__(compose(config_name="fit_cfm", overrides=[*overrides, *extra]))
        return ad.read_zarr(tmp_path / "data.zarr")

    first = run()
    assert (tmp_path / "run" / "config.json").is_file()
    assert first.obsm["X_pca"].shape == (first.n_obs, 20) and "pca" in first.uns["preprocessing"]
    assert set(first.obs["split"]) == {"train", "val", "test", "all"}

    # A second run on the same file reuses its split, whatever the seed; only the PCA needs an explicit overwrite.
    second = run("run.splitter_seed=7", "preprocessing.overwrite=true", "run.allow_overwrite=true")
    assert second.obs["split"].equals(first.obs["split"])


def test_zarr_input_reads_only_what_the_run_uses(tmp_path: Path):
    obs = {"drug1": ["control", "a"], "drug2": ["control", "b"], "cell_line": ["x", "x"], "unused": ["u", "v"]}
    uns = {"drug": {"a": np.ones(2)}, "cell_line": {"x": np.ones(2)}, "big_unused_table": np.ones(1000)}
    ad.AnnData(np.ones((2, 3), dtype=np.float32), obs=obs, uns=uns, obsm={"X_other": np.ones((2, 2))}).write_zarr(
        tmp_path / "d.zarr"
    )
    with initialize_config_dir(config_dir=str(Path(fit_cfm.__file__).parent / "configs"), version_base=None):
        cfg = compose(config_name="fit_cfm", overrides=["+experiment=cellflow"])
    adata = fit_cfm.read_zarr_input(tmp_path / "d.zarr", cfg)
    assert set(adata.obs.columns) == {"drug1", "drug2", "cell_line"}
    assert set(adata.uns) == {"drug", "cell_line"}
    assert not isinstance(adata.obsm["X_other"], np.ndarray)  # left on disk


def test_cellflow_smoke_on_controls_and_perturbed_in_separate_files(tmp_path: Path):
    adata = get_dummy_adata(
        n_obs_pert=400,
        n_obs_ctrl=100,
        n_genes=60,
        obsm_keys_to_dim={},
        obs_columns_to_nunique_and_prefix={"drug1": (3, "drug"), "drug2": (3, "drug"), "cell_line": (2, "cell_line")},
        uns_keys_to_nunique_prefix_and_dim={"drug": (3, "drug", 8), "cell_line": (2, "cell_line", 4)},
        obs_columns_to_fixed_val={"drug1": "control", "drug2": "control"},
        rng=np.random.default_rng(1),
    )
    adata.X = sp.csr_matrix(np.log1p(np.abs(adata.X)).astype(np.float32))
    adata.obs_names = adata.obs_names.astype(str)
    is_ctrl = (adata.obs["drug1"] == "control").to_numpy()
    paths = [tmp_path / "ctrl.zarr", tmp_path / "pert.zarr"]
    adata[is_ctrl].copy().write_zarr(paths[0])
    adata[~is_ctrl].copy().write_zarr(paths[1])

    overrides = [
        "+experiment=cellflow_smoke",
        f"paths.target_adata=[{paths[0]},{paths[1]}]",
        f"save.run_dir={tmp_path / 'run'}",
        f"callbacks.model_checkpoint.dirpath={tmp_path / 'ckpt'}",
        f"loggers.wandb.save_dir={tmp_path / 'wandb'}",
        f"loggers.csv.save_dir={tmp_path / 'csv'}",
        f"trainer.default_root_dir={tmp_path / 'logs'}",
        "trainer.fast_dev_run=true",
        "trainer.enable_progress_bar=false",
        "splitter.test_fraction=0.3",
        "splitter.val_fraction=0.3",
        "hydra.job.chdir=false",
    ]

    def run(*extra: str) -> None:
        with initialize_config_dir(config_dir=str(Path(fit_cfm.__file__).parent / "configs"), version_base=None):
            fit_cfm.main.__wrapped__(compose(config_name="fit_cfm", overrides=[*overrides, *extra]))

    run()
    ctrl, pert = (ad.read_zarr(p) for p in paths)
    assert set(ctrl.obs["split"]) == {"all"} and {"train", "val", "test"} <= set(pert.obs["split"])
    assert ctrl.obsm["X_pca"].shape == (ctrl.n_obs, 20) and pert.obsm["X_pca"].shape == (pert.n_obs, 20)
    np.testing.assert_array_equal(  # one basis for both files
        ctrl.uns["preprocessing"]["pca"]["PCs"], pert.uns["preprocessing"]["pca"]["PCs"]
    )

    run("run.splitter_seed=7", "preprocessing.overwrite=true", "run.allow_overwrite=true")  # the split is reused
    assert ad.read_zarr(paths[1]).obs["split"].equals(pert.obs["split"])

    no_split = ad.read_zarr(paths[0])  # the controls' file loses its split; the other keeps it
    del no_split.obs["split"]
    no_split.write_zarr(paths[0])
    with pytest.raises(ValueError, match=r"is in some input files but not in \['ctrl.zarr'\]"):
        run("preprocessing.overwrite=true", "run.allow_overwrite=true")


def test_cellflow_flows_from_the_controls_and_genot_from_noise():
    with initialize_config_dir(config_dir=str(Path(fit_cfm.__file__).parent / "configs"), version_base=None):
        cellflow = compose(config_name="fit_cfm", overrides=["+experiment=cellflow"])
        genot = compose(config_name="fit_cfm", overrides=["+experiment=[cellflow,genot]"])
    assert not cellflow.training.generate_from_noise and not cellflow.inference.generate_from_noise
    assert genot.training.generate_from_noise and genot.module.source_encoder_mlp_kwargs is not None


def test_one_out_dir_places_every_output():
    with initialize_config_dir(config_dir=str(Path(fit_cfm.__file__).parent / "configs"), version_base=None):
        cfg = compose(config_name="fit_cfm", overrides=["paths.out_dir=/runs/x"], return_hydra_config=True)
    assert (
        cfg.save.run_dir,
        cfg.callbacks.model_checkpoint.dirpath,
        cfg.trainer.default_root_dir,
        cfg.loggers.wandb.save_dir,
        cfg.loggers.csv.save_dir,
        cfg.hydra.run.dir,
    ) == (
        "/runs/x/save",
        "/runs/x/checkpoints",
        "/runs/x/logs",
        "/runs/x/logs/wandb",
        "/runs/x/logs/csv",
        "/runs/x/hydra",
    )
