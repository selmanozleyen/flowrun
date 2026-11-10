"""End to end on a toy screen: prepare, fit both flows and the classifier, guide."""

import json
import shutil
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from flowrun import fit_cfm, labcompass

COLUMNS = labcompass.DOSES + labcompass.LINEAR


def _screen(n_designs: int = 8, cells_per_design: int = 40) -> ad.AnnData:
    """A toy screen in which pairs of designs share a dose recipe and differ in culture conditions.

    The recipe of toy designs 0 and 1 is the most MgK-rich, so it is the target's, and only it shows "Rare" cells.
    """
    rng = np.random.default_rng(0)
    levels = rng.uniform(0, 10, size=(n_designs, len(COLUMNS))).round(1)
    levels[1::2, : len(labcompass.DOSES)] = levels[::2, : len(labcompass.DOSES)]
    design = np.repeat(np.arange(n_designs), cells_per_design)
    types = np.array(["MgK_a", "MgK_b", "Ery", "Mono", None, "Rare"], dtype=object)  # None: unassigned cells
    picked = np.where(design < 2, rng.choice([0, 1, 5], len(design)), rng.integers(0, 5, len(design)))
    obs = pd.DataFrame(levels[design], columns=COLUMNS)
    obs["cell_type"] = pd.Categorical(types[picked])
    obs.index = obs.index.astype(str)
    X = (rng.normal(size=(len(design), 6)) + levels[design, :6] / 10).astype(np.float32)
    return ad.AnnData(X=X, obs=obs)


def _prepare(screen: ad.AnnData, out: Path) -> dict:
    screen.write_h5ad(out.with_suffix(".h5ad"))
    return labcompass.prepare(out.with_suffix(".h5ad"), out, target_prefix="MgK", n_val=2, seed=0)


def _read(out: Path) -> tuple[ad.AnnData, ad.AnnData]:
    return ad.read_zarr(out / "cells.zarr"), ad.read_zarr(out / "designs.zarr")


def test_prepare_fits_nothing_on_held_out_rows(tmp_path: Path):
    """Moving the markers of every val and test cell, and the target recipe's doses, changes no fitted statistic."""
    screen = _screen()
    _prepare(screen, tmp_path / "base")
    base, base_designs = _read(tmp_path / "base")
    train = (base.obs["split"] == "train").to_numpy()
    test = (base.obs["split"] == "test").to_numpy()
    screen.X[~train] += 100.0
    # both designs of the recipe move together, so they stay one recipe; the first dose column, which orders the
    # designs, is kept so that every design keeps its id
    screen.obs.loc[test, labcompass.DOSES[1:]] += 100.0
    _prepare(screen, tmp_path / "moved")
    moved, moved_designs = _read(tmp_path / "moved")

    np.testing.assert_array_equal(moved.obs["design_id"], base.obs["design_id"])
    np.testing.assert_array_equal(moved.obs["split"], base.obs["split"])
    assert not np.array_equal(moved_designs.obs[COLUMNS], base_designs.obs[COLUMNS])
    np.testing.assert_array_equal(moved.X[train], base.X[train])
    for stat in ("mean", "std"):
        np.testing.assert_array_equal(
            moved.uns["preprocessing"]["zscore"][stat], base.uns["preprocessing"]["zscore"][stat]
        )
    moved_encoding, base_encoding = (json.loads(a.uns["design_encoding"]) for a in (moved, base))
    for stat in ("mean", "std", "min", "max"):
        assert moved_encoding[stat] == base_encoding[stat]
    # and directly: the encoding is fit on the training designs, not on val ones
    designs = ad.read_zarr(tmp_path / "base" / "designs.zarr").obs
    train = designs.loc[designs["split"] == "train", COLUMNS].to_numpy(float)
    logged = np.where(base_encoding["log1p"], np.log1p(train), train)
    np.testing.assert_allclose(base_encoding["mean"], logged.mean(0))


def test_prepare_holds_out_the_target_recipe(tmp_path: Path):
    """The target is the most MgK-rich design; it and every design with its doses are "test", nothing else is."""
    target = str(_prepare(_screen(), tmp_path / "data")["target_design"])
    cells, designs = _read(tmp_path / "data")
    assert target == designs.obs["target_fraction"].idxmax()
    doses = designs.obs[labcompass.DOSES]
    same_recipe = (doses == doses.loc[target]).all(axis=1)
    assert same_recipe.sum() == 2  # the toy screen pairs designs by recipe
    np.testing.assert_array_equal(designs.obs["split"] == "test", same_recipe)
    np.testing.assert_array_equal(
        cells.obs["split"] == "test", cells.obs["design_id"].isin(designs.obs_names[same_recipe])
    )

    labcompass._check_split(cells, designs, target)
    split = cells.obs["split"].astype(str).to_numpy()
    split[0] = "val" if split[0] == "train" else "train"
    cells.obs["split"] = split
    with pytest.raises(ValueError, match="not the one `prepare` wrote"):
        labcompass._check_split(cells, designs, target)


def test_encoding_round_trips(tmp_path: Path):
    _prepare(_screen(), tmp_path / "data")
    _, designs = _read(tmp_path / "data")
    decoded = labcompass.decode(designs.X, json.loads(designs.uns["design_encoding"]))
    np.testing.assert_allclose(decoded[COLUMNS], designs.obs[COLUMNS], atol=1e-4)


def test_composition_is_over_all_cells():
    """Types outside the classes keep their share of the denominator and are reported as unknown mass."""
    ystar, unknown = labcompass._composition(pd.Series(["a", "a", "b", "x"]), ["a", "b", "c"])
    np.testing.assert_allclose(ystar, [0.5, 0.25, 0.0])
    assert unknown == pytest.approx(0.25)


class _Shifted:
    """A response whose every population is a design's first coordinate plus ``shift``."""

    def __init__(self, shift: float):
        self.shift = shift

    def population_responses(self, cond_dict: dict, n_populations: int) -> torch.Tensor:
        x = cond_dict[labcompass.DESIGN_KEY]
        return (x[:, :1] + self.shift).expand(n_populations, 1, len(x), 1)


def test_best_is_picked_on_scoring_and_reported_on_fresh():
    def potential(ys, ystar):
        return (ys - ystar).squeeze(-1) ** 2

    x, ystar = torch.tensor([[0.0], [2.0], [1.0]]), torch.tensor([1.0])
    pot, _ = labcompass._score(_Shifted(0.0), x, potential, ystar)
    torch.testing.assert_close(pot, torch.tensor([1.0, 1.0, 0.0]))
    # the best design is picked on the scoring potential and reported on the fresh response
    assert labcompass._best(_Shifted(5.0), x, pot, potential, ystar) == 25.0


def _fit(experiment: str, data: Path, out: Path) -> Path:
    overrides = [
        f"+experiment={experiment}",
        f"paths.target_adata={data}",
        f"paths.out_dir={out}",
        "trainer.fast_dev_run=true",
        "trainer.enable_progress_bar=false",
    ]
    with initialize_config_dir(config_dir=str(Path(fit_cfm.__file__).parent / "configs"), version_base=None):
        fit_cfm.main.__wrapped__(compose(config_name="fit_cfm", overrides=overrides))
    return out / "save"


@pytest.fixture(scope="module")
def trained(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, dict]:
    """The toy screen prepared, with a classifier and one-step surrogate and prior fits."""
    root = tmp_path_factory.mktemp("labcompass")
    _prepare(_screen(), root / "data")
    fitted = labcompass.fit_classifier(root / "data", hidden=8, epochs=1, lr=1e-3, seed=0)
    _fit("labcompass_surrogate", root / "data" / "cells.zarr", root / "surrogate")
    _fit("labcompass_prior", root / "data" / "designs.zarr", root / "prior")
    return root, fitted


def _guide(root: Path, out: Path, **kwargs) -> dict:
    return labcompass.guide(root / "data", root / "surrogate" / "save", root / "prior" / "save", out, **kwargs)


def test_fit_classifier(trained, tmp_path: Path):
    """It learns from training cells only: "Rare" only shows in the held-out target recipe, and moving the held-out
    cells' markers leaves the fitted weights unchanged."""
    root, fitted = trained
    model, classes = labcompass.load_classifier(root / "data", "cpu")
    assert classes == ["Ery", "MgK_a", "MgK_b", "Mono", labcompass.UNASSIGNED]
    assert fitted["n_classes"] == len(classes)
    assert all(0 <= fitted[f"{s}_accuracy"] <= 1 for s in ("train", "val", "test"))

    cells = ad.read_zarr(root / "data" / "cells.zarr")
    cells.X[(cells.obs["split"] != "train").to_numpy()] += 100.0
    cells.write_zarr(tmp_path / "cells.zarr")
    labcompass.fit_classifier(tmp_path, hidden=8, epochs=1, lr=1e-3, seed=0)
    moved, _ = labcompass.load_classifier(tmp_path, "cpu")
    for (name, a), (_, b) in zip(model.state_dict().items(), moved.state_dict().items(), strict=True):
        torch.testing.assert_close(a, b, msg=name)


@pytest.mark.parametrize("readout", ["classifier", "markers"])
def test_guide(trained, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, readout: str):
    from sckitflow.core.methods.inference import ODEInference

    root, _ = trained
    cells, designs = _read(root / "data")

    # the surrogate must generate `n_cells` cells per design for every response, not one
    drawn, predict = [], ODEInference.predict

    def spy(self, step_data, *, generator):
        pred = predict(self, step_data, generator=generator)
        drawn.append(pred.raw_samples.shape[0])
        return pred

    monkeypatch.setattr(ODEInference, "predict", spy)  # the prior is sampled with ODESolver, not `predict`
    # and which response each step of the summary uses: guiding has seed 0, scoring 1 and fresh 2
    seeds, score, best = {"score": set(), "best": set()}, labcompass._score, labcompass._best
    monkeypatch.setattr(labcompass, "_score", lambda r, *a: seeds["score"].add(r.seed) or score(r, *a))
    monkeypatch.setattr(labcompass, "_best", lambda r, *a: seeds["best"].add(r.seed) or best(r, *a))
    args = {"n_designs": 4, "n_steps": 3, "scale": 1.0, "max_strength": 10.0, "n_cells": 5, "surrogate_steps": 2}
    summary = _guide(root, tmp_path, readout=readout, seed=0, **args)

    assert drawn and set(drawn) == {5}
    assert seeds == {"score": {1, 2}, "best": {2}}  # `_best` scores its pick on the fresh response
    assert (tmp_path / "summary.json").exists()
    assert summary["args"] == {**args, "readout": readout, "seed": 0}
    assert summary["target_observed_fraction"] == designs.obs["target_fraction"].max()
    assert 0 <= summary["target_observed_fraction"] <= 1 and np.isfinite(summary["target_predicted_potential"])
    if readout == "classifier":  # the target's "Rare" cells are a type the classifier never saw
        on_target = cells.obs["design_id"] == str(summary["target_design"])
        rare = (cells.obs.loc[on_target, "cell_type"] == "Rare").mean()
        assert rare > 0 and summary["target_unknown_mass"] == pytest.approx(rare)
    assert summary["train"]["n_designs"] == (designs.obs["split"] == "train").sum()
    for kind in ("train", "prior", "guided"):
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


def test_guide_refuses_a_foreign_split(trained, tmp_path: Path):
    root, _ = trained
    shutil.copytree(root / "data", tmp_path / "data")
    designs = ad.read_zarr(tmp_path / "data" / "designs.zarr")
    designs.obs["split"] = "train"  # as if the prior had re-split the designs
    labcompass._write(designs, tmp_path / "data" / "designs.zarr")
    with pytest.raises(ValueError, match="not the one `prepare` wrote"):
        labcompass.guide(tmp_path / "data", root / "surrogate" / "save", root / "prior" / "save", tmp_path,
                         n_designs=1, n_steps=None, scale=1.0, max_strength=1.0, n_cells=1, surrogate_steps=None,
                         readout="markers", seed=0)  # fmt: skip


@pytest.mark.parametrize("overwrite", [True, None], ids=["resplit", "no-config"])
def test_guide_refuses_an_unverified_run(trained, tmp_path: Path, overwrite: bool | None):
    """A prior trained with ``splitter.overwrite=true``, or one whose run config is gone, may have seen the target."""
    root, _ = trained
    prior = tmp_path / "prior"  # guide checks the run configs before loading anything, so the config is enough
    prior.mkdir()
    if overwrite is not None:
        config = OmegaConf.load(root / "prior" / "save" / "config_resolved.yaml")
        config.splitter.overwrite = overwrite
        OmegaConf.save(config, prior / "config_resolved.yaml")
    with pytest.raises((ValueError, FileNotFoundError), match="splitter.overwrite=true|is missing"):
        labcompass.guide(root / "data", root / "surrogate" / "save", prior, tmp_path / "out", n_designs=1,
                         n_steps=None, scale=1.0, max_strength=1.0, n_cells=1, surrogate_steps=None,
                         readout="markers", seed=0)  # fmt: skip


@pytest.mark.parametrize(("n_steps", "n_cells"), [(1, 1), (None, 0)])
def test_guide_rejects_bad_sizes(tmp_path: Path, n_steps: int | None, n_cells: int):
    with pytest.raises(ValueError, match="n_steps|n_cells"):
        _guide(tmp_path, tmp_path, n_designs=1, n_steps=n_steps, scale=1.0, max_strength=1.0, n_cells=n_cells,
               surrogate_steps=None, readout="markers", seed=0)  # fmt: skip


def test_cli_prepare(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture):
    _screen().write_h5ad(tmp_path / "screen.h5ad")
    monkeypatch.setattr(
        "sys.argv",
        ["flowrun-labcompass", "prepare", str(tmp_path / "screen.h5ad"), str(tmp_path / "out"), "--n-val", "1"],
    )
    labcompass.main()
    assert len(json.loads(capsys.readouterr().out)["val_designs"]) == 1
