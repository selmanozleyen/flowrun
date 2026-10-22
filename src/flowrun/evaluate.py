"""Per-cell evaluation of a saved panel model: ``python -m flowrun.evaluate <save dir>``.

No cell is measured on both panels, so each score uses a different stand-in for ground truth, for source cells
(the control panel) the model maps to the target panel:

- shared-marker consistency: a cell's predicted shared markers against its own measured ones, each panel
  standardised on its own cells (the panels read the same marker differently);
- neighbourhood preservation: how much of a cell's neighbourhood in its own panel stays its neighbourhood among
  the predictions.

Every score comes with two baselines without learning: each source cell's nearest target cell on the shared
markers, and a random target cell. Writes ``evaluation.json`` into the save dir.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import anndata as ad
import numpy as np
import torch
from omegaconf import OmegaConf
from sklearn.neighbors import NearestNeighbors

from flowrun.panels import _state_values, from_state


def _standardise(x: np.ndarray, reference: np.ndarray) -> np.ndarray:
    return (x - reference.mean(0)) / (reference.std(0) + 1e-8)


def _rowwise_corr(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a, b = a - a.mean(1, keepdims=True), b - b.mean(1, keepdims=True)
    return (a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-8)


def shared_consistency(predicted: np.ndarray, measured: np.ndarray) -> dict[str, float]:
    """Predicted against measured shared markers, both standardised: per-marker r, per-cell r, RMSE."""
    per_marker = _rowwise_corr(predicted.T, measured.T)
    return {
        "shared_r_per_marker": float(per_marker.mean()),
        "shared_r_per_cell": float(_rowwise_corr(predicted, measured).mean()),
        "shared_rmse": float(np.sqrt(((predicted - measured) ** 2).mean())),
    }


def neighbourhood_preservation(source: np.ndarray, predicted: np.ndarray, k: int = 15) -> float:
    """Mean fraction of each cell's k nearest neighbours (own panel) that stay its neighbours among the predictions."""
    own = NearestNeighbors(n_neighbors=k + 1).fit(source).kneighbors(return_distance=False)[:, 1:]
    new = NearestNeighbors(n_neighbors=k + 1).fit(predicted).kneighbors(return_distance=False)[:, 1:]
    return float(np.mean([len(set(a) & set(b)) / k for a, b in zip(own, new, strict=True)]))


def predict_from_sources(model, datamodule, adatas, source_state: np.ndarray, *, seed: int, batch: int = 2048):
    """The model's prediction (mean over its samples) for each given source state, in the target state's space.

    Takes the eval loader's batch for the target group (its condition encodings, as sckitflow builds them) and puts
    the given source cells in, so every prediction is that cell's.
    """
    step_data, _ = next(iter(datamodule.dm.get_eval_loader(adatas, max_per_group=1)))
    method = model.inference_method
    device = next(method.module.parameters()).device
    out = []
    for start in range(0, len(source_state), batch):
        rows = torch.as_tensor(source_state[start : start + batch], dtype=torch.float32, device=device)
        data = {**step_data, "source_state": rows, "n_obs": len(rows), "target_state": None}
        generator = torch.Generator(device=device).manual_seed(seed + start)
        with torch.no_grad():
            out.append(method.predict(data, generator=generator).X.detach().cpu().numpy())
    return np.concatenate(out)


def evaluate(save_dir: Path, *, n_cells: int = 5000, k: int = 15, seed: int = 0) -> dict:
    from sckitflow import Model
    from sckitflow.data import FlowDataModule

    from flowrun.fit_cfm import as_list, read_zarr_input

    cfg = OmegaConf.load(save_dir / "config_resolved.yaml")
    paths = [Path(p) for p in as_list(cfg.paths.target_adata)]
    adatas = [read_zarr_input(p, cfg) if p.suffix == ".zarr" else ad.read_h5ad(p) for p in paths]
    from_state(adatas, OmegaConf.to_container(cfg.datamanager, resolve=True))
    model = Model.from_pretrained(save_dir, adatas)
    model.module.to("cuda" if torch.cuda.is_available() else "cpu")
    datamodule = FlowDataModule.from_adata(adatas, model.manager, n_train_steps=1, batch_size=256)

    # one file per panel; the control panel's file holds the sources
    key, columns = next(iter(cfg.datamanager.categorical_conditions.items()))
    control = str(cfg.datamanager.control_values_dict[key])
    is_source = [str(a.obs[columns[0]].iloc[0]) == control for a in adatas]
    source, target = adatas[is_source.index(True)], adatas[is_source.index(False)]
    state = cfg.datamanager.target_sample_rep
    source_all, target_all = _state_values(source, state), _state_values(target, state)

    rng = np.random.default_rng(seed)
    rows = np.sort(rng.choice(source.n_obs, size=min(n_cells, source.n_obs), replace=False))
    predicted = predict_from_sources(model, datamodule, adatas, source_all[rows], seed=seed)

    shared = sorted(set(source.var_names) & set(target.var_names))
    s_idx = [list(source.var_names).index(m) for m in shared]
    t_idx = [list(target.var_names).index(m) for m in shared]
    source_z = _standardise(source_all[rows], source_all)
    measured_shared = source_z[:, s_idx]

    def scores(prediction: np.ndarray) -> dict[str, float]:
        z = _standardise(prediction, target_all)
        return {
            **shared_consistency(z[:, t_idx], measured_shared),
            "neighbourhood_preservation": neighbourhood_preservation(source_z, z, k=k),
        }

    target_z = _standardise(target_all, target_all)
    nearest = NearestNeighbors(n_neighbors=1).fit(target_z[:, t_idx]).kneighbors(measured_shared, return_distance=False)
    result = {
        "model": scores(predicted),
        "baseline_nearest_on_shared": scores(target_all[nearest[:, 0]]),
        "baseline_random": scores(target_all[rng.choice(target.n_obs, size=len(rows))]),
        "n_cells": len(rows),
        "k": k,
        "shared": shared,
    }
    (save_dir / "evaluation.json").write_text(json.dumps(result, indent=2))
    return result


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Per-cell evaluation of a saved panel model (see flowrun.evaluate).")
    parser.add_argument("save_dir", type=Path, help="The run's save/ folder (config_resolved.yaml and the model).")
    parser.add_argument("--n-cells", type=int, default=5000)
    parser.add_argument("--k", type=int, default=15)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    result = evaluate(args.save_dir, n_cells=args.n_cells, k=args.k, seed=args.seed)
    keys = [k for k in result["model"]]
    print(f"{'':28s}" + "".join(f"{k:>28s}" for k in keys))
    for name in ("model", "baseline_nearest_on_shared", "baseline_random"):
        print(f"{name:28s}" + "".join(f"{result[name][k]:28.3f}" for k in keys))


if __name__ == "__main__":
    main()
