"""End to end on a toy screen: prepare, fit both flows and the classifier, guide."""

import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from hydra import compose, initialize_config_dir

from flowrun import fit_cfm, labcompass


def _screen(n_designs: int = 8, cells_per_design: int = 40) -> ad.AnnData:
    """A toy screen in which pairs of designs share a dose recipe and differ in culture conditions."""
    rng = np.random.default_rng(0)
    columns = labcompass.DOSES + labcompass.LINEAR
    levels = rng.uniform(0, 10, size=(n_designs, len(columns))).round(1)
    levels[1::2, : len(labcompass.DOSES)] = levels[::2, : len(labcompass.DOSES)]
    design = np.repeat(np.arange(n_designs), cells_per_design)
    types = np.array(["MgK_a", "MgK_b", "Ery", "Mono", None], dtype=object)
    obs = pd.DataFrame(levels[design], columns=columns)
    obs["cell_type"] = pd.Categorical(types[rng.integers(0, len(types), len(design))])  # None: unassigned cells
    obs.index = obs.index.astype(str)
    X = (rng.normal(size=(len(design), 6)) + levels[design, :6] / 10).astype(np.float32)
    return ad.AnnData(X=X, obs=obs)


def _prepare(screen: ad.AnnData, out: Path) -> dict:
    screen.write_h5ad(out.with_suffix(".h5ad"))
    return labcompass.prepare(out.with_suffix(".h5ad"), out, target_prefix="MgK", n_val=2, seed=0)


def test_prepare_fits_nothing_on_held_out_rows(tmp_path: Path):
    """Moving the markers of every val and test cell leaves the training cells and every fitted statistic as is."""
    screen = _screen()
    _prepare(screen, tmp_path / "base")
    base = ad.read_zarr(tmp_path / "base" / "cells.zarr")
    train = (base.obs["split"] == "train").to_numpy()
    screen.X[~train] += 100.0
    _prepare(screen, tmp_path / "moved")
    moved = ad.read_zarr(tmp_path / "moved" / "cells.zarr")

    np.testing.assert_array_equal(moved.obs["split"], base.obs["split"])
    np.testing.assert_array_equal(moved.X[train], base.X[train])
    np.testing.assert_array_equal(
        moved.uns["preprocessing"]["zscore"]["mean"], base.uns["preprocessing"]["zscore"]["mean"]
    )
    np.testing.assert_array_equal(
        moved.uns["preprocessing"]["zscore"]["std"], base.uns["preprocessing"]["zscore"]["std"]
    )
    assert moved.uns["design_encoding"] == base.uns["design_encoding"]


def _fit(experiment: str, data: Path, run_dir: Path, tmp_path: Path) -> None:
    overrides = [
        f"+experiment={experiment}",
        f"paths.target_adata={data}",
        f"save.run_dir={run_dir}",
        f"callbacks.model_checkpoint.dirpath={tmp_path / experiment / 'ckpt'}",
        f"loggers.wandb.save_dir={tmp_path / experiment / 'wandb'}",
        f"loggers.csv.save_dir={tmp_path / experiment / 'csv'}",
        f"trainer.default_root_dir={tmp_path / experiment / 'logs'}",
        "trainer.fast_dev_run=true",
        "trainer.enable_progress_bar=false",
        "hydra.job.chdir=false",
    ]
    with initialize_config_dir(config_dir=str(Path(fit_cfm.__file__).parent / "configs"), version_base=None):
        fit_cfm.main.__wrapped__(compose(config_name="fit_cfm", overrides=overrides))


@pytest.fixture(scope="module")
def trained(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, dict]:
    """The toy screen prepared, with a classifier and one-step surrogate and prior fits."""
    root = tmp_path_factory.mktemp("labcompass")
    data = root / "data"
    prepared = _prepare(_screen(), data)
    labcompass.fit_classifier(data, hidden=8, epochs=1, lr=1e-3, seed=0)
    _fit("labcompass_surrogate", data / "cells.zarr", root / "surrogate", root)
    _fit("labcompass_prior", data / "designs.zarr", root / "prior", root)
    return root, prepared


COLUMNS = labcompass.DOSES + labcompass.LINEAR


def test_prepare_holds_out_the_target_recipe(trained):
    """The target and every design with its doses are "test" in both files, and nothing else is."""
    root, prepared = trained
    designs, cells = ad.read_zarr(root / "data" / "designs.zarr"), ad.read_zarr(root / "data" / "cells.zarr")
    doses = designs.obs[labcompass.DOSES]
    same_recipe = (doses == doses.loc[str(prepared["target_design"])]).all(axis=1)
    assert same_recipe.sum() == 2  # the toy screen pairs designs by recipe
    np.testing.assert_array_equal(designs.obs["split"] == "test", same_recipe)
    np.testing.assert_array_equal(
        cells.obs["split"] == "test", cells.obs["design_id"].isin(designs.obs_names[same_recipe])
    )


@pytest.mark.parametrize("readout", ["classifier", "markers"])
def test_guide(trained, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, readout: str):
    from sckitflow.core.methods.inference._ode import ODEInference

    root, prepared = trained
    designs = ad.read_zarr(root / "data" / "designs.zarr")

    # the surrogate must generate `n_cells` cells per design for every response, not one
    drawn, predict = [], ODEInference.predict

    def spy(self, step_data, *, generator):
        pred = predict(self, step_data, generator=generator)
        drawn.append(pred.raw_samples.shape[0])
        return pred

    monkeypatch.setattr(ODEInference, "predict", spy)  # the prior is sampled with ODESolver, not `predict`
    kwargs = {"n_designs": 4, "n_steps": 3, "scale": 1.0, "max_strength": 10.0, "n_cells": 5, "surrogate_steps": 2}
    summary = labcompass.guide(
        root / "data", root / "surrogate", root / "prior", tmp_path, readout=readout, seed=0, **kwargs
    )

    assert drawn and set(drawn) == {5}
    assert json.loads((tmp_path / "summary.json").read_text()) == summary
    for kind in ("prior", "guided"):
        fraction = summary[kind].pop("predicted_target_fraction")
        assert all(np.isfinite(v) for v in summary[kind].values())
        assert (fraction is None) == (readout == "markers") and (fraction is None or 0 <= fraction <= 1)

    proposed = pd.read_csv(tmp_path / "designs.csv")
    assert proposed["kind"].value_counts().to_dict() == {"prior": 4, "guided": 4, "target": 1}
    # one-step toy models barely move the designs, so this only checks that guidance acted at all
    potentials = {k: proposed.loc[proposed["kind"] == k, "potential"].to_numpy() for k in ("prior", "guided")}
    assert not np.array_equal(potentials["prior"], potentials["guided"])
    # sampled designs are clipped to the training designs' range before they are scored and written
    train = designs.obs.loc[designs.obs["split"] == "train", COLUMNS].to_numpy(float)
    sampled = proposed.loc[proposed["kind"] != "target", COLUMNS].to_numpy(float)
    assert np.all(sampled >= train.min(0) - 1e-4) and np.all(sampled <= train.max(0) + 1e-4)
    target = proposed.loc[proposed["kind"] == "target", COLUMNS].to_numpy(float)[0]
    np.testing.assert_allclose(target, designs.obs.loc[str(prepared["target_design"]), COLUMNS].to_numpy(float))
