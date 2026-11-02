"""LabCompass inverse design: a surrogate of cells given a culture design, a flow prior over designs, and
loss-guided sampling of designs whose predicted cells match a target.

1. ``flowrun-labcompass prepare``: encodes the designs (log1p doses, z-scored over designs) into
   ``obsm["design"]`` of the cells and into ``X`` of a one-row-per-design file, z-scores the markers, and writes
   a split that holds the target design out of both.
2. ``flowrun-fit-cfm +experiment=labcompass_surrogate`` and ``+experiment=labcompass_prior`` train the two flows.
3. ``flowrun-labcompass guide``: samples designs from the prior steered toward cells whose mean marker profile
   matches the target design's (``sckitflow.core.surrogate.LossGuidedVelocity``) and reports them next to
   unguided prior samples.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

DESIGN_KEY = "design"
DOSES = [
    "gm-csf_[ng_ml]",
    "tpo_[ng_ml]",
    "sr1_[nm]",
    "um171_[nm]",
    "um729_[µm]",
    "scf_[ng_ml]",
    "butyzamide_[nm]",
    "retinoic_acid_[µm]",
    "ldl_[ng_ml]",
    "il3_[ng_ml]",
]
LINEAR = ["o2_[%]", "days_of_culture"]


def prepare(adata_path: Path, out_dir: Path, *, target_prefix: str, n_val: int, seed: int) -> dict:
    """Writes ``cells.zarr`` and ``designs.zarr`` to ``out_dir`` and returns the encoding summary."""
    adata = ad.read_h5ad(adata_path, backed="r")
    obs = adata.obs
    raw = pd.DataFrame({c: pd.to_numeric(obs[c].astype(str)) for c in DOSES + LINEAR}, index=obs.index)
    design_id = raw.groupby(DOSES + LINEAR, sort=True).ngroup().to_numpy()

    designs = raw.groupby(design_id).first()
    encoded = np.concatenate([np.log1p(designs[DOSES].to_numpy()), designs[LINEAR].to_numpy()], axis=1)
    mean, std = encoded.mean(0), encoded.std(0) + 1e-8
    encoded = ((encoded - mean) / std).astype(np.float32)

    cell_type = obs["cell_type"].astype(str).to_numpy()
    fraction = pd.Series(cell_type).str.startswith(target_prefix).groupby(design_id).mean()
    target = int(fraction.idxmax())
    rng = np.random.default_rng(seed)
    val = rng.choice(np.setdiff1d(designs.index, [target]), size=n_val, replace=False)
    split = pd.Series("train", index=designs.index)
    split[val], split[target] = "val", "test"

    X = np.asarray(adata.X[:], dtype=np.float32)
    marker_mean, marker_std = X.mean(0), X.std(0) + 1e-8
    encoding = {
        "columns": DOSES + LINEAR,
        "log1p": [True] * len(DOSES) + [False] * len(LINEAR),
        "mean": mean.tolist(),
        "std": std.tolist(),
        "marker_mean": marker_mean.tolist(),
        "marker_std": marker_std.tolist(),
        "target_design": target,
        "target_prefix": target_prefix,
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    cells = ad.AnnData(
        X=(X - marker_mean) / marker_std,
        obs=pd.DataFrame(
            {
                "design_id": pd.Categorical(design_id.astype(str)),
                "cell_type": pd.Categorical(cell_type),
                "split": pd.Categorical(split.to_numpy()[design_id]),
            },
            index=obs.index.astype(str),
        ),
        var=pd.DataFrame(index=adata.var_names.astype(str)),
        obsm={DESIGN_KEY: encoded[design_id]},
        uns={"design_encoding": json.dumps(encoding)},
    )
    cells.write_zarr(out_dir / "cells.zarr")
    design_obs = designs.rename(columns=str).assign(target_fraction=fraction, split=pd.Categorical(split.to_numpy()))
    design_obs.index = design_obs.index.astype(str)
    ad.AnnData(
        X=encoded,
        obs=design_obs,
        var=pd.DataFrame(index=[f"enc_{c}" for c in DOSES + LINEAR]),
        uns={"design_encoding": json.dumps(encoding)},
    ).write_zarr(out_dir / "designs.zarr")
    return {
        "n_cells": cells.n_obs,
        "n_designs": len(designs),
        "target_design": target,
        "target_fraction": float(fraction[target]),
        "max_train_fraction": float(fraction[split == "train"].max()),
        "val_designs": sorted(int(v) for v in val),
    }


def decode(encoded: np.ndarray, encoding: dict) -> pd.DataFrame:
    """Maps encoded designs back to doses (negative doses are clipped to 0)."""
    x = encoded * np.asarray(encoding["std"]) + np.asarray(encoding["mean"])
    x = np.where(encoding["log1p"], np.expm1(x).clip(min=0), x)
    return pd.DataFrame(x, columns=encoding["columns"])


def guide(
    data_dir: Path,
    surrogate_dir: Path,
    prior_dir: Path,
    out_dir: Path,
    *,
    n_designs: int,
    n_steps: int,
    scale: float,
    max_strength: float,
    n_context: int,
    surrogate_steps: int,
    seed: int,
) -> dict:
    """Samples guided and unguided designs and writes ``designs.csv`` and ``summary.json`` to ``out_dir``."""
    import torch
    from sckitflow import Model
    from sckitflow.core.solvers import ODESolver
    from sckitflow.core.surrogate import (
        LossGuidedVelocity,
        MeanSamplesReadoutConfig,
        SquaredErrorPotentialConfig,
        SurrogatePotential,
    )
    from sckitflow.data import FlowDataModule

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cells, designs = ad.read_zarr(data_dir / "cells.zarr"), ad.read_zarr(data_dir / "designs.zarr")
    encoding = json.loads(cells.uns["design_encoding"])
    target = str(encoding["target_design"])

    surrogate = Model.from_pretrained(surrogate_dir, cells)
    prior = Model.from_pretrained(prior_dir, designs)
    for model in (surrogate, prior):
        model.module.to(device).eval().requires_grad_(False)
    inferer = surrogate.inference_method
    inferer.config = inferer.config.model_copy(update={"n_steps": surrogate_steps})

    # The potential reads its context (one leaf, n_context cells) from training cells; only their count matters
    # when generating from noise. The target is the held-out design's mean marker profile.
    rng = np.random.default_rng(seed)
    context = cells[rng.choice(np.flatnonzero(cells.obs["split"] == "train"), n_context, replace=False)].copy()
    ystar = torch.as_tensor(cells.X[(cells.obs["design_id"] == target).to_numpy()].mean(0), device=device)
    potential = SurrogatePotential(
        inferer=inferer,
        ystar=ystar,
        datamodule=FlowDataModule.from_adata(context, surrogate.manager, n_train_steps=1, batch_size=n_context),
        adata=context,
        readout=MeanSamplesReadoutConfig().build(),
        potential=SquaredErrorPotentialConfig().build(),
        require_target_state=False,
        seed=seed,
    ).to(device)

    def psi(x: torch.Tensor) -> torch.Tensor:
        return potential({DESIGN_KEY: x})

    path = prior.inference_method.probability_path
    x0 = torch.randn(n_designs, designs.n_vars, generator=torch.Generator().manual_seed(seed)).to(device)
    time = torch.linspace(0, 1, n_steps, device=device)
    samples = {}
    for name, s in (("prior", 0.0), ("guided", scale)):
        field = LossGuidedVelocity(prior.module, path, psi, scale=s, max_strength=max_strength)
        samples[name] = ODESolver(field, device_id=device).solve(x0, time).detach()

    target_x = torch.as_tensor(designs[target].X[0], device=device)
    train_x = torch.as_tensor(designs[designs.obs["split"] == "train"].X, device=device)
    rows, summary = [], {"target_design": int(target), "scale": scale, "n_designs": n_designs}
    with torch.no_grad():
        summary["target_potential"] = float(psi(target_x[None])[0])
        summary["train_potential_min"] = float(psi(train_x).min())
    for name, x in samples.items():
        with torch.no_grad():
            pot = psi(x)
        dist = torch.linalg.vector_norm(x - target_x, dim=-1)
        nearest = torch.cdist(x, train_x).min(-1).values
        summary[name] = {
            "potential_mean": float(pot.mean()),
            "potential_min": float(pot.min()),
            "dist_to_target_mean": float(dist.mean()),
            "dist_to_target_min": float(dist.min()),
            "dist_to_nearest_train_mean": float(nearest.mean()),
        }
        doses = decode(x.cpu().numpy(), encoding).assign(
            kind=name, potential=pot.cpu().numpy(), dist_to_target=dist.cpu().numpy()
        )
        rows.append(doses)
    rows.append(decode(target_x[None].cpu().numpy(), encoding).assign(kind="target"))

    out_dir.mkdir(parents=True, exist_ok=True)
    pd.concat(rows, ignore_index=True).to_csv(out_dir / "designs.csv", index=False)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare", help="encode designs and markers, write cells.zarr and designs.zarr")
    p.add_argument("adata", type=Path)
    p.add_argument("out_dir", type=Path)
    p.add_argument("--target-prefix", default="MgK", help="cell types whose fraction picks the target design")
    p.add_argument("--n-val", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    g = sub.add_parser("guide", help="sample guided and prior designs")
    g.add_argument("data_dir", type=Path)
    g.add_argument("surrogate_dir", type=Path)
    g.add_argument("prior_dir", type=Path)
    g.add_argument("out_dir", type=Path)
    g.add_argument("--n-designs", type=int, default=64)
    g.add_argument("--n-steps", type=int, default=50)
    g.add_argument("--scale", type=float, default=1.0)
    g.add_argument("--max-strength", type=float, default=10.0)
    g.add_argument("--n-context", type=int, default=256, help="cells generated per design by the surrogate")
    g.add_argument("--surrogate-steps", type=int, default=20, help="ODE steps of the surrogate inside guidance")
    g.add_argument("--seed", type=int, default=0)
    args = vars(parser.parse_args())
    command = args.pop("command")
    if command == "prepare":
        result = prepare(args.pop("adata"), args.pop("out_dir"), **args)
    else:
        result = guide(
            args.pop("data_dir"), args.pop("surrogate_dir"), args.pop("prior_dir"), args.pop("out_dir"), **args
        )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
