"""Read only what a tool needs from an AnnData on disk.

A tool names the parts of an AnnData it touches with :mod:`anndata.acc` references, and :func:`open_adata` opens a
``.zarr`` or ``.h5ad`` store with exactly those parts:

    from anndata.acc import A

    cells = open_adata("cells.zarr", A.X, A.obsm["design"], A.obs["split"], A.obs["cell_type"], uns=["encoding"])
    target_x = read_rows(cells.X, cells.obs["split"] == "test")

Arrays (``A.X``, ``A.layers[k]``, ``A.obsm[k]``) of a ``.zarr`` store stay on disk until indexed, so reading a few
rows costs a few rows (:func:`read_rows`). Those of a ``.h5ad`` file are read into memory and the file is closed: an
open h5py handle blocks writes to the file (``fit_cfm`` writes its split back) and doesn't survive a forked data
loader worker. Only the referenced arrays are read either way. ``obs`` columns (``A.obs[col]``) and ``uns`` entries are read into memory one by one, so an
unused column of a wide ``obs`` is never decoded. ``A.obs.index`` reads the obs names; without it the frame gets a
``RangeIndex``, since on large stores the index is a big string array most tools never use. A reference the store
lacks raises, naming what the store has. A reference into an array, such as ``A.obsm["pca"][:, 0]``, opens the
whole array.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from anndata.acc import A, AdRef, LayerAcc, MetaAcc, MultiAcc
from scipy import sparse

__all__ = ["keys", "open_adata", "open_inputs", "read_rows", "resolve"]

type Ref = AdRef | LayerAcc | MultiAcc


def resolve(spec: str) -> Ref:
    """A reference from a config string: ``"X"``, ``"layers/<k>"``, ``"obsm/<k>"`` or an :meth:`A.resolve` form.

    Configs name arrays as ``"obsm/X_pca"``; anndata's own string form is ``"obsm.X_pca"``. Both are accepted.
    """
    return A.resolve(spec.replace("/", ".", 1), vec=False) if "[" not in spec else A.resolve(spec)


def _locate(ref: Ref) -> tuple[str, str | None]:
    """``("rep", "X" | "layers/<k>" | "obsm/<k>")``, ``("obs", <column>)`` or ``("index", None)``."""
    acc = ref.acc if isinstance(ref, AdRef) else ref
    match acc:
        case LayerAcc(k=None):
            return "rep", "X"
        case LayerAcc(k=k):
            return "rep", f"layers/{k}"
        case MultiAcc(dim="obs", k=k):
            return "rep", f"obsm/{k}"
        case MetaAcc(dim="obs") if isinstance(ref, AdRef):
            return ("index", None) if ref.idx is None else ("obs", ref.idx)
    raise ValueError(f"{ref!r}: only X, layers, obsm and obs columns or index can be opened.")


def _open_group(path: Path):
    if path.suffix == ".zarr":
        import zarr

        return zarr.open_group(path, mode="r")
    if path.suffix == ".h5ad":
        import h5py

        return h5py.File(path, "r")
    raise ValueError(f"{path}: expected a .zarr or .h5ad store.")


def _keys(group) -> list[str]:
    return sorted(group.keys()) if group is not None else []


def keys(path: str | Path, kind: str) -> list[str]:
    """The keys a store holds under ``kind`` (``"obsm"``, ``"layers"``, ``"uns"``, ...), without reading them."""
    group = _open_group(Path(path))
    try:
        return _keys(group.get(kind))
    finally:
        if hasattr(group, "close"):  # h5py: an open handle would block reopening the file
            group.close()


def _check(path: Path, group, reps: set[str], columns: set[str], uns: set[str]) -> None:
    obs = group["obs"]
    obs_columns = set(obs.attrs.get("column-order", [])) or set(_keys(obs)) - {obs.attrs.get("_index")}
    missing = []
    for rep in sorted(reps):
        if rep == "X":
            if "X" not in group:
                missing.append("X")
            continue
        kind, key = rep.split("/", 1)
        if key not in _keys(group.get(kind)):
            missing.append(f"{kind}[{key!r}] (has {kind}: {_keys(group.get(kind))})")
    if cols := sorted(columns - obs_columns):
        missing.append(f"obs {cols} (has {sorted(obs_columns)})")
    if keys := sorted(uns - set(_keys(group.get("uns")))):
        missing.append(f"uns {keys} (has {_keys(group.get('uns'))})")
    if missing:
        raise KeyError(f"{path} lacks " + "; ".join(missing))


def open_adata(path: str | Path, *refs: Ref, uns: Iterable[str] = ()) -> ad.AnnData:
    """Open ``path`` with exactly the referenced parts: arrays on disk, ``obs`` columns and ``uns`` entries in memory.

    ``var`` is reduced to its index. Raises :class:`KeyError` naming every missing part and what the store has.
    """
    from scfit.data._io import read_backing

    located = [_locate(r) for r in refs]
    reps = {name for kind, name in located if kind == "rep"}
    columns = {name for kind, name in located if kind == "obs"}
    uns = set(uns)
    path = Path(path)
    group = _open_group(path)
    _check(path, group, reps, columns, uns)

    obs_group, var_group = group["obs"], group["var"]
    arrays = {rep: read_backing(group[rep]) for rep in reps}
    h5ad = path.suffix == ".h5ad"
    if h5ad:  # read the requested arrays and close: an open h5py handle blocks writes and doesn't survive a fork
        arrays = {rep: a.to_memory() if hasattr(a, "to_memory") else a[...] for rep, a in arrays.items()}
    obs = pd.DataFrame({c: ad.io.read_elem(obs_group[c]) for c in sorted(columns)})
    if ("index", None) in located or not (arrays or columns):  # with nothing else to size it by, read the index
        obs.index = pd.Index(ad.io.read_elem(obs_group[obs_group.attrs["_index"]])).astype(str)
    else:
        obs.index = pd.RangeIndex(next(iter(arrays.values())).shape[0] if arrays else len(obs)).astype(str)
    var = pd.DataFrame(index=pd.Index(ad.io.read_elem(var_group[var_group.attrs["_index"]])).astype(str))

    adata = ad.AnnData(X=arrays.pop("X", None), obs=obs, var=var)
    for rep, array in arrays.items():
        kind, key = rep.split("/", 1)
        getattr(adata, kind)[key] = array
    for key in sorted(uns):
        adata.uns[key] = ad.io.read_elem(group["uns"][key])
    if h5ad:
        group.close()
    return adata


def open_inputs(paths: str | Path | Sequence[str | Path], *refs: Ref, uns: Iterable[str] = ()) -> list[ad.AnnData]:
    """:func:`open_adata` for one path or a list of them, as a list."""
    paths = [paths] if isinstance(paths, str | Path) else list(paths)
    return [open_adata(p, *refs, uns=uns) for p in paths]


def read_rows(array, rows: np.ndarray | pd.Series | slice | None = None) -> np.ndarray:
    """Rows of an on-disk array as a dense numpy array: all of them, a slice, a boolean mask or integer positions.

    Positions are read in sorted order, which zarr and h5py read fastest, and returned in the order given.
    """
    if rows is None:
        rows = slice(None)
    if isinstance(rows, slice):
        out = array[rows]
    else:
        rows = np.asarray(rows)
        if rows.dtype == bool:
            rows = np.flatnonzero(rows)
        order = np.argsort(rows, kind="stable")
        out = array[rows[order]]
        out = out[np.argsort(order, kind="stable")]
    return np.asarray(out.toarray() if sparse.issparse(out) else out)
