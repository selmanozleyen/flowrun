"""End to end on a toy screen: prepare, fit both flows and the classifier, guide."""

import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from hydra import compose, initialize_config_dir

from flowrun import fit_cfm, labcompass


def _screen(path: Path, n_designs: int = 8, cells_per_design: int = 40) -> None:
    rng = np.random.default_rng(0)
    columns = labcompass.DOSES + labcompass.LINEAR
    levels = rng.uniform(0, 10, size=(n_designs, len(columns))).round(1)
    design = np.repeat(np.arange(n_designs), cells_per_design)
    types = np.array(["MgK_a", "Ery", "Mono", None], dtype=object)
    obs = pd.DataFrame(levels[design], columns=columns)
    obs["cell_type"] = pd.Categorical(types[rng.integers(0, len(types), len(design))])  # None: unassigned cells
    obs.index = obs.index.astype(str)
    X = (rng.normal(size=(len(design), 6)) + levels[design, :6] / 10).astype(np.float32)
    ad.AnnData(X=X, obs=obs).write_h5ad(path)


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


def test_guide_end_to_end(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from sckitflow.core.methods.inference._ode import ODEInference

    _screen(tmp_path / "screen.h5ad")
    data = tmp_path / "data"
    labcompass.prepare(tmp_path / "screen.h5ad", data, target_prefix="MgK", n_val=2, seed=0)
    labcompass.fit_classifier(data, hidden=8, epochs=1, lr=1e-3, seed=0)
    _fit("labcompass_surrogate", data / "cells.zarr", tmp_path / "surrogate", tmp_path)
    _fit("labcompass_prior", data / "designs.zarr", tmp_path / "prior", tmp_path)

    # every surrogate prediction must average `n_cells` generated cells per design, not one
    drawn = []
    predict = ODEInference.predict

    def spy(self, step_data, *, generator):
        pred = predict(self, step_data, generator=generator)
        if pred.raw_samples is not None and "design" in step_data["target_condition_data"]:
            drawn.append(pred.raw_samples.shape[0])
        return pred

    monkeypatch.setattr(ODEInference, "predict", spy)
    kwargs = {"n_designs": 4, "n_steps": 3, "scale": 1.0, "max_strength": 10.0, "n_cells": 5, "surrogate_steps": 2}
    summary = labcompass.guide(
        data, tmp_path / "surrogate", tmp_path / "prior", tmp_path / "guide", readout="classifier", seed=0, **kwargs
    )

    assert drawn and set(drawn) == {5}
    assert json.loads((tmp_path / "guide" / "summary.json").read_text()) == summary
    for kind in ("prior", "guided"):
        assert all(np.isfinite(v) for v in summary[kind].values())
        assert 0 <= summary[kind]["predicted_target_type_fraction"] <= 1
    designs = pd.read_csv(tmp_path / "guide" / "designs.csv")
    assert designs["kind"].value_counts().to_dict() == {"prior": 4, "guided": 4, "target": 1}
