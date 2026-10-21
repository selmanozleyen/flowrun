"""Cells measured on different marker panels, one file per panel.

:func:`prepare_panels` (``python -m flowrun.panels``) writes each panel to zarr with its markers renamed (``rename``,
for synonyms such as ``EPCR`` / ``CD201``; the original names stay in ``var["original_name"]``), its label in
``obs[label_key]``, the labels' one-hot table in ``uns[label_key]`` (sorted label order) and these steps in
``uns["panels"]``. No split and no transform: ``fit_cfm`` splits and writes ``obs["split_<seed>"]`` back.

:func:`panel_reps` runs in ``fit_cfm`` (``panels.layer``): from one layer, as is, each file gets in memory

- ``obsm["X_state"]``: all of its markers, the state;
- ``obsm["X_lin"]``: the markers every file measures, by name, sorted, the coupling's shared term;
- ``obsm["X_quad"]``: its own markers, sorted, the coupling's quadratic term.

These follow Lorenzo's ``prepare_panel_adata_nosplit.py`` (theislab/sckitflow#202), plus the renames and the state.

    python -m flowrun.panels --out /path/to/dir --rename CD41a=CD41 EPCR=CD201 \\
        --panel "old panel" /path/to/old.h5ad --panel "new panel" /path/to/new.h5ad
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd


def prepare_panels(
    adatas: Sequence[ad.AnnData], *, labels: Sequence[str], rename: Mapping[str, str], label_key: str = "dataset"
) -> None:
    """Renames each panel's markers and labels it, in place (see the module)."""
    if len(labels) != len(adatas):
        raise ValueError(f"{len(adatas)} panels but {len(labels)} labels; give one label per panel.")
    onehot = {label: np.eye(len(labels), dtype=np.float32)[i] for i, label in enumerate(sorted(labels))}
    for a, label in zip(adatas, labels, strict=True):
        names = [rename.get(v, v) for v in a.var_names]
        if len(set(names)) < len(names):
            raise ValueError(f"Panel {label!r} has duplicate markers after renaming: {sorted(names)}.")
        a.var["original_name"] = a.var_names.to_numpy()
        a.var_names = names
        a.obs[label_key] = pd.Categorical([label] * a.n_obs, categories=sorted(labels))
        a.uns[label_key] = onehot
        a.uns["panels"] = {"panel": label, "label_key": label_key, "rename": dict(rename)}


def panel_reps(adatas: Sequence[ad.AnnData], *, layer: str) -> dict:
    """Adds each file's state, shared and own markers from ``layers[layer]``, in place (see the module)."""
    names = [list(a.var_names) for a in adatas]
    shared = sorted(set(names[0]).intersection(*names[1:]))
    if not shared:
        raise ValueError("The panels share no marker; map synonyms with `python -m flowrun.panels --rename`.")
    own = []
    for a, n in zip(adatas, names, strict=True):
        values = np.asarray(a.layers[layer][:], dtype=np.float32)
        col = {v: i for i, v in enumerate(n)}
        own.append(sorted(set(n) - set(shared)))
        a.obsm["X_state"] = values
        a.obsm["X_lin"] = values[:, [col[v] for v in shared]]
        a.obsm["X_quad"] = values[:, [col[v] for v in own[-1]]]
    return {"layer": layer, "shared": shared, "own": own}


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Prepare one zarr per marker panel (see flowrun.panels).")
    parser.add_argument("--panel", nargs=2, action="append", required=True, metavar=("LABEL", "PATH"))
    parser.add_argument("--out", type=Path, required=True, help="Directory for the <input stem>.zarr outputs.")
    parser.add_argument("--label-key", default="dataset")
    parser.add_argument("--rename", nargs="*", default=[], metavar="OLD=NEW")
    args = parser.parse_args(argv)

    labels, paths = zip(*((label, Path(p)) for label, p in args.panel), strict=True)
    adatas = [ad.read_zarr(p) if p.suffix == ".zarr" else ad.read_h5ad(p) for p in paths]
    prepare_panels(adatas, labels=labels, rename=dict(r.split("=", 1) for r in args.rename), label_key=args.label_key)
    shared = sorted(set(adatas[0].var_names).intersection(*(a.var_names for a in adatas[1:])))
    print(f"{len(shared)} shared markers: {shared}")
    args.out.mkdir(parents=True, exist_ok=True)
    for a, label, p in zip(adatas, labels, paths, strict=True):
        out = args.out / f"{p.stem}.zarr"
        a.write_zarr(out)
        print(f"{label}: {a.n_obs} cells, own markers {sorted(set(a.var_names) - set(shared))} -> {out}")


if __name__ == "__main__":
    main()
