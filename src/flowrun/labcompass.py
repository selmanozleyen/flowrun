"""LabCompass inverse design: a surrogate of cells given a culture design, a flow prior over designs, and
loss-guided sampling of designs whose predicted cells match a target.

1. ``flowrun-labcompass prepare``: encodes the designs (log1p doses, z-scored over designs) into
   ``obsm["design"]`` of the cells and into ``X`` of a one-row-per-design file, z-scores the markers, and writes
   a split that holds the target design out of both.
2. ``flowrun-fit-cfm +experiment=labcompass_surrogate`` and ``+experiment=labcompass_prior`` train the two flows.
3. ``flowrun-labcompass fit-classifier``: the readout ``h``, a differentiable cell-type classifier on the markers.
4. ``flowrun-labcompass guide``: samples designs from the prior steered toward cells whose predicted cell-type
   composition matches the target design's (``sckitflow.core.surrogate.LossGuidedVelocity``) and reports them
   next to unguided prior samples.

Everything is retrospective: the designs are scored by the trained surrogate against a held-out design of an
existing screen, never carried out.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse

from flowrun.preprocess import preprocess

DESIGN_KEY = "design"
UNASSIGNED = "unassigned"  # cells the screen left without a type: a class of its own, so ȳ(x) still sums to 1
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


def _write(adata: ad.AnnData, path: Path) -> None:
    """Writes ``adata`` in the row-chunked sharded layout annbatch streams."""
    import zarr
    from annbatch import write_sharded

    write_sharded(zarr.open_group(path, mode="w", zarr_format=3), adata)


def prepare(adata_path: Path, out_dir: Path, *, target_prefix: str, n_val: int, seed: int) -> dict:
    """Writes ``cells.zarr`` and ``designs.zarr`` to ``out_dir`` and returns the encoding summary.

    The target is the design with the largest fraction of ``target_prefix`` cell types. It and every design with
    the same doses, the same recipe at another day or O2 level, are held out as "test", so a prior or surrogate
    never trained on its recipe. ``n_val`` other designs are "val". Every normalisation is fit on the training rows.
    """
    adata = ad.read_h5ad(adata_path, backed="r")
    obs = adata.obs
    raw = pd.DataFrame({c: pd.to_numeric(obs[c].astype(str), errors="coerce") for c in DOSES + LINEAR}, index=obs.index)
    if missing := [c for c in raw if raw[c].isna().any()]:
        raise ValueError(f"Design columns with missing or non-numeric values: {missing}.")
    design_id = raw.groupby(DOSES + LINEAR, sort=True).ngroup().to_numpy()
    designs = raw.groupby(design_id).first()

    cell_type = obs["cell_type"].astype(object).fillna(UNASSIGNED).astype(str).to_numpy()
    fraction = pd.Series(cell_type).str.startswith(target_prefix).groupby(design_id).mean()
    target = int(fraction.idxmax())
    same_recipe = (designs[DOSES] == designs.loc[target, DOSES]).all(axis=1)
    if n_val > (~same_recipe).sum() - 1:
        raise ValueError(f"n_val={n_val} leaves no training design: only {(~same_recipe).sum()} are not held out.")
    rng = np.random.default_rng(seed)
    val = rng.choice(designs.index[~same_recipe], size=n_val, replace=False)
    split = pd.Series("train", index=designs.index)
    split[val], split[same_recipe] = "val", "test"
    is_train = (split == "train").to_numpy()
    cell_split = split.to_numpy()[design_id]

    logged = np.concatenate([np.log1p(designs[DOSES].to_numpy()), designs[LINEAR].to_numpy()], axis=1)
    mean, std = logged[is_train].mean(0), logged[is_train].std(0)
    std[std == 0] = 1  # a column constant over the training designs is only centred
    encoding = {
        "columns": DOSES + LINEAR,
        "log1p": [True] * len(DOSES) + [False] * len(LINEAR),
        "mean": mean.tolist(),
        "std": std.tolist(),
        "min": designs[is_train].min().tolist(),
        "max": designs[is_train].max().tolist(),
        "target_design": target,
        "target_prefix": target_prefix,
    }
    encoded = encode(designs.to_numpy(), encoding)

    X = adata.X[:]
    X = np.asarray(X.toarray() if sparse.issparse(X) else X, dtype=np.float32)
    var_names = adata.var_names.astype(str)
    adata.file.close()

    out_dir.mkdir(parents=True, exist_ok=True)
    cells = ad.AnnData(
        X=X,
        obs=pd.DataFrame(
            {
                "design_id": pd.Categorical(design_id.astype(str)),
                "cell_type": pd.Categorical(cell_type),
                "split": pd.Categorical(cell_split),
            },
            index=obs.index.astype(str),
        ),
        var=pd.DataFrame(index=var_names),
        obsm={DESIGN_KEY: encoded[design_id]},
        uns={"design_encoding": json.dumps(encoding)},
    )
    # z-scored on the training cells; `uns["preprocessing"]` maps generated cells back with `preprocess.inverse`
    preprocess(cells, train_mask=cell_split == "train", zscore=True, key_added="zscored")
    cells.X = cells.obsm.pop("zscored").astype(np.float32)
    _write(cells, out_dir / "cells.zarr")
    design_obs = designs.assign(target_fraction=fraction, split=pd.Categorical(split.to_numpy()))
    design_obs.index = design_obs.index.astype(str)
    designs_adata = ad.AnnData(
        X=encoded,
        obs=design_obs,
        var=pd.DataFrame(index=[f"enc_{c}" for c in DOSES + LINEAR]),
        uns={"design_encoding": json.dumps(encoding)},
    )
    _write(designs_adata, out_dir / "designs.zarr")
    return {
        "n_cells": cells.n_obs,
        "n_designs": len(designs),
        "target_design": target,
        "target_fraction": float(fraction[target]),
        "max_train_fraction": float(fraction[split == "train"].max()),
        "test_designs": sorted(int(d) for d in designs.index[same_recipe]),
        "val_designs": sorted(int(v) for v in val),
    }


def fit_classifier(data_dir: Path, *, hidden: int, epochs: int, lr: float, seed: int) -> dict:
    """Trains the cell-type readout ``h`` on the training cells and writes ``classifier.pt`` to ``data_dir``.

    The paper's readout maps a generated cell to its type; guidance needs it differentiable, so it is a small MLP
    on the z-scored markers rather than the Leiden clustering it is trained against.
    """
    import torch

    cells = ad.read_zarr(data_dir / "cells.zarr")
    labels = cells.obs["cell_type"].astype(str)  # `prepare` already named the unassigned cells
    split = cells.obs["split"].to_numpy()
    classes = sorted(labels[split == "train"].unique())  # a type only held-out designs show is not learnable
    y_all = torch.as_tensor(pd.Index(classes).get_indexer(labels))  # -1 for such a type
    X_all = torch.as_tensor(np.asarray(cells.X, dtype=np.float32))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(seed)
    model = _classifier_module(cells.n_vars, len(classes), hidden).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    X, y = X_all[split == "train"].to(device), y_all[split == "train"].to(device)
    generator = torch.Generator(device=device).manual_seed(seed)
    for _ in range(epochs):
        perm = torch.randperm(len(X), device=device, generator=generator)
        for i in range(0, len(X), 4096):
            idx = perm[i : i + 4096]
            loss = torch.nn.functional.cross_entropy(model(X[idx]), y[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
    model.eval()
    accuracy = {}
    with torch.no_grad():
        for name in ("train", "val", "test"):
            rows = (split == name) & (y_all.numpy() >= 0)
            Xs, ys = X_all[rows].to(device), y_all[rows].to(device)
            accuracy[f"{name}_accuracy"] = float((model(Xs).argmax(-1) == ys).float().mean()) if len(ys) else None
    torch.save(
        {"state_dict": model.state_dict(), "classes": classes, "hidden": hidden, "n_markers": cells.n_vars},
        data_dir / "classifier.pt",
    )
    return {"n_classes": len(classes), "epochs": epochs, **accuracy}


def _classifier_module(n_markers: int, n_classes: int, hidden: int):
    import torch

    return torch.nn.Sequential(
        torch.nn.Linear(n_markers, hidden),
        torch.nn.SiLU(),
        torch.nn.Linear(hidden, hidden),
        torch.nn.SiLU(),
        torch.nn.Linear(hidden, n_classes),
    )


def load_classifier(data_dir: Path, device: str):
    """The trained readout and its class names."""
    import torch

    state = torch.load(data_dir / "classifier.pt", map_location=device, weights_only=True)
    model = _classifier_module(state["n_markers"], len(state["classes"]), state["hidden"])
    model.load_state_dict(state["state_dict"])
    return model.to(device).eval().requires_grad_(False), state["classes"]


def encode(doses: np.ndarray, encoding: dict) -> np.ndarray:
    """Maps doses to the encoded space: log1p of the dose columns, then z-scored on the training designs."""
    x = np.where(encoding["log1p"], np.log1p(doses), doses)
    return ((x - np.asarray(encoding["mean"])) / np.asarray(encoding["std"])).astype(np.float32)


def decode(encoded: np.ndarray, encoding: dict) -> pd.DataFrame:
    """Maps encoded designs back to doses; the inverse of `encode`."""
    x = encoded * np.asarray(encoding["std"]) + np.asarray(encoding["mean"])
    return pd.DataFrame(np.where(encoding["log1p"], np.expm1(x), x), columns=encoding["columns"])


SCORE_DRAWS = 4  # independent populations each design is scored on, apart from the guiding one


def guide(
    data_dir: Path,
    surrogate_dir: Path,
    prior_dir: Path,
    out_dir: Path,
    *,
    n_designs: int,
    n_steps: int | None,
    scale: float,
    max_strength: float,
    n_cells: int,
    surrogate_steps: int | None,
    readout: str,
    seed: int,
) -> dict:
    """Samples guided and unguided designs and writes ``designs.csv`` and ``summary.json`` to ``out_dir``.

    The surrogate generates ``n_cells`` cells per design to predict its response. The designs are guided with
    one draw of those cells and scored on ``SCORE_DRAWS`` others, so the summary does not reward fitting the
    guiding noise. Sampled designs are clipped to the training designs' range before they are scored, so each
    row of ``designs.csv`` pairs its doses with their own potential. A "best" design is picked on the scoring
    draws and its potential reported on ``SCORE_DRAWS`` fresh ones, so sets of different sizes compare fairly.
    Distances are in the encoded space (z-scored log doses). ``n_steps`` and ``surrogate_steps`` default to the
    trained models' own inference steps.
    """
    import torch
    from sckitflow import Model
    from sckitflow.core.solvers import ODESolver
    from sckitflow.core.surrogate import (
        LossGuidedVelocity,
        PathFlowMap,
        SurrogatePotential,
        SurrogateResponse,
        cross_entropy_probs,
        squared_error,
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
    # Generating from noise, the surrogate draws `n_samples` cells per design. The context only supplies one leaf;
    # its own design is replaced by the optimized one, so a single training cell is enough.
    update = {"n_samples": n_cells} | ({} if surrogate_steps is None else {"n_steps": surrogate_steps})
    inferer.config = inferer.config.model_copy(update=update)
    context = cells[np.flatnonzero(cells.obs["split"] == "train")[:1]].copy()
    is_target = (cells.obs["design_id"] == target).to_numpy()
    target_classes = None
    if readout == "classifier":
        classifier, classes = load_classifier(data_dir, device)
        # ``ȳ*``: the target design's observed composition, from the cell types the classifier was trained on.
        counts = cells.obs.loc[is_target, "cell_type"].astype(str).value_counts()
        ystar = torch.tensor(
            (counts.reindex(classes).fillna(0) / is_target.sum()).to_numpy(), dtype=torch.float32, device=device
        )
        target_classes = torch.as_tensor([c.startswith(encoding["target_prefix"]) for c in classes], device=device)
        readout_fn, potential_fn = (lambda c: classifier(c).softmax(-1)), cross_entropy_probs
    else:
        ystar = torch.tensor(np.asarray(cells.X[is_target]).mean(0), device=device)
        readout_fn, potential_fn = None, squared_error
    response = SurrogateResponse(
        inferer=inferer,
        datamodule=FlowDataModule.from_adata(context, surrogate.manager),
        adata=context,
        readout=readout_fn,
        require_target_state=False,
        seed=seed,
    )
    potential = SurrogatePotential(response, ystar=ystar, potential=potential_fn).to(device)

    def psi(x: torch.Tensor) -> torch.Tensor:
        return potential({DESIGN_KEY: x})

    def score(x: torch.Tensor, fresh: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
        """Each design's potential and response, averaged over the context's groups and ``SCORE_DRAWS`` draws.

        Population 0 is the guiding draw, 1 to ``SCORE_DRAWS`` the scoring ones and the next ``SCORE_DRAWS``
        the fresh ones. The skipped populations are still generated: ``population_responses`` has no offset.
        """
        first = SCORE_DRAWS + 1 if fresh else 1
        with torch.no_grad():
            ys = response.population_responses({DESIGN_KEY: x}, first + SCORE_DRAWS)[first:]
        return potential_fn(ys, ystar).mean((0, 1)), ys.mean((0, 1))

    def best(x: torch.Tensor, pot: torch.Tensor) -> float:
        """The fresh-draw potential of the design with the lowest scoring-draw potential."""
        return float(score(x[pot.argmin()][None], fresh=True)[0][0])

    lo, hi = (torch.tensor(encode(np.asarray(encoding[k])[None], encoding)[0], device=device) for k in ("min", "max"))

    path = prior.inference_method.probability_path
    x0 = prior.inference_method.noise_sampler(
        (n_designs, designs.n_vars),
        generator=torch.Generator(device=device).manual_seed(seed),
        device=device,
        dtype=torch.float32,
    )
    time = torch.linspace(0, 1, prior.inference_method.config.n_steps if n_steps is None else n_steps, device=device)
    method = prior.inference_method.config.solver_kwargs.get("method", "euler")
    guided = LossGuidedVelocity(
        prior.module, path, psi, flow_map=PathFlowMap(prior.module, path), scale=scale, max_strength=max_strength
    )
    samples = {
        name: ODESolver(field, method=method, device_id=device).solve(x0, time).detach().clamp(lo, hi)
        for name, field in (("prior", prior.module), ("guided", guided))
    }

    target_x = torch.as_tensor(designs[target].X[0], device=device)
    train_x = torch.as_tensor(designs[designs.obs["split"] == "train"].X, device=device)
    rows, summary = [], {"target_design": int(target), "scale": scale, "n_designs": n_designs, "readout": readout}
    if target_classes is not None:  # the paper's headline number: the mass on the target's cell types
        summary["target_prefix"] = encoding["target_prefix"]
        summary["target_observed_fraction"] = float(ystar[target_classes].sum())

    summary["target_predicted_potential"] = float(score(target_x[None])[0][0])
    summary["train_potential_best"] = best(train_x, score(train_x)[0])
    for name, x in samples.items():
        pot, y = score(x)
        dist = torch.linalg.vector_norm(x - target_x, dim=-1)
        nearest = torch.cdist(x, train_x).min(-1).values
        summary[name] = {
            "potential_mean": float(pot.mean()),
            "potential_best": best(x, pot),
            "predicted_target_fraction": None
            if target_classes is None
            else float(y[..., target_classes].sum(-1).mean()),
            "dist_to_target_mean": float(dist.mean()),
            "dist_to_target_min": float(dist.min()),
            "dist_to_nearest_train_mean": float(nearest.mean()),
        }
        doses = decode(x.cpu().numpy(), encoding).assign(
            kind=name, potential=pot.cpu().numpy(), dist_to_target=dist.cpu().numpy()
        )
        rows.append(doses)
    rows.append(designs.obs.loc[[target], encoding["columns"]].reset_index(drop=True).assign(kind="target"))

    out_dir.mkdir(parents=True, exist_ok=True)
    pd.concat(rows, ignore_index=True).to_csv(out_dir / "designs.csv", index=False)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(required=True)
    p = sub.add_parser("prepare", help="encode designs and markers, write cells.zarr and designs.zarr")
    p.set_defaults(func=prepare)
    p.add_argument("adata_path", type=Path)
    p.add_argument("out_dir", type=Path)
    p.add_argument("--target-prefix", default="MgK", help="cell types whose fraction picks the target design")
    p.add_argument("--n-val", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    c = sub.add_parser("fit-classifier", help="train the cell-type readout on the prepared cells")
    c.set_defaults(func=fit_classifier)
    c.add_argument("data_dir", type=Path)
    c.add_argument("--hidden", type=int, default=128)
    c.add_argument("--epochs", type=int, default=10)
    c.add_argument("--lr", type=float, default=1e-3)
    c.add_argument("--seed", type=int, default=0)
    g = sub.add_parser("guide", help="sample guided and prior designs")
    g.set_defaults(func=guide)
    g.add_argument("data_dir", type=Path)
    g.add_argument("surrogate_dir", type=Path)
    g.add_argument("prior_dir", type=Path)
    g.add_argument("out_dir", type=Path)
    g.add_argument("--n-designs", type=int, default=64)
    g.add_argument("--n-steps", type=int, help="time points of the prior's ODE; defaults to the prior's own")
    g.add_argument("--scale", type=float, default=1.0)
    g.add_argument("--max-strength", type=float, default=10.0)
    g.add_argument("--n-cells", type=int, default=256, help="cells the surrogate generates per design")
    g.add_argument("--surrogate-steps", type=int, help="ODE steps of the surrogate; defaults to its own")
    g.add_argument(
        "--readout",
        choices=("classifier", "markers"),
        default="classifier",
        help="classifier: cell-type composition against the target's, scored by cross-entropy (the paper's "
        "potential, needs `fit-classifier`); markers: mean marker profile, squared error",
    )
    g.add_argument("--seed", type=int, default=0)
    args = vars(parser.parse_args())
    print(json.dumps(args.pop("func")(**args), indent=2))


if __name__ == "__main__":
    main()
