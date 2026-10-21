"""Preprocessing fitted on the training cells and stored with the data, so the transformation is known.

In order, each optional: PCA (:func:`scanpy.pp.pca` on dask, from ``X``), z-score, whitening. Without PCA the other
steps start from an existing representation (``obsm[input_key]``) or ``X``. Every step is fitted on the training rows
only and applied to every row, lazily on dask, so a backed zarr input is read chunk by chunk and held-out cells never
shape the result. The output goes to ``obsm[key_added]``; ``uns["preprocessing"]`` records every fitted step, so
:func:`apply` repeats the transformation on new data and :func:`inverse` maps predictions back to the input features.

TODO(scale): PCA and whitening build a features x features covariance, so ``n_hvg=None`` on ~20k genes needs ~3 GB
and minutes of ``eigh``. For larger data: default to HVGs (as cf-train does), and add a GPU path (rapids-singlecell,
or cf-train's cupy loop).
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Literal

import anndata as ad
import dask.array as da
import numpy as np
import scanpy as sc
import scipy.linalg
import zarr
from anndata import AnnData
from fast_array_utils import stats
from scipy.sparse import issparse

from flowrun._store import open_store

WhiteningMethod = Literal["zca", "pca", "cholesky"]


def preprocess(
    adata: AnnData | Sequence[AnnData],
    *,
    train_mask: np.ndarray | None = None,
    pca: int | None = None,
    n_hvg: int | None = None,
    hvg_flavor: str = "seurat",
    zscore: bool = False,
    whitening: WhiteningMethod | None = None,
    epsilon: float = 1e-8,
    input_key: str | None = None,
    key_added: str = "X_pca",
    chunk_size: int = 10_000,
) -> None:
    """Fit the steps on the training rows and transform every row, in place.

    Parameters
    ----------
    adata
        One AnnData or a list read as one dataset, rows in list order: the steps are fitted over all of them, and
        each gets its own rows' output.
    train_mask
        Boolean mask over the rows (of all files, in list order) the steps are fitted on; ``None`` fits on every row.
        Pass the training split, so held-out cells never shape the space they are scored in.
    pca
        Number of principal components of ``X``; ``None`` skips PCA.
    n_hvg
        Fit the PCA on this many highly variable genes; ``None`` uses every gene.
    hvg_flavor
        Flavor for :func:`scanpy.pp.highly_variable_genes`. ``"seurat"`` expects log-normalised data.
    zscore
        Center and scale every feature to unit variance.
    whitening
        Decorrelate the features to unit covariance: ``"zca"`` (closest to the input), ``"pca"`` (rotated onto
        the principal axes) or ``"cholesky"``; ``None`` skips it.
    epsilon
        Added to the variances before dividing by them.
    input_key
        Where the steps start without PCA: ``obsm[input_key]``, or ``X`` when ``None``. PCA always starts from ``X``.
    key_added
        Key of the output in ``obsm``.
    chunk_size
        Rows per dask chunk.

    Writes ``obsm[key_added]`` and ``uns["preprocessing"]`` (the step order, the input, and each step's fitted
    parameters) into every AnnData.
    """
    adatas = [adata] if isinstance(adata, AnnData) else list(adata)
    source = None if pca is not None else input_key
    for i, other in enumerate(adatas[1:], start=1):
        if source is None and not other.var_names.equals(adatas[0].var_names):
            raise ValueError(
                f"AnnData {i} has different genes (var_names) than AnnData 0, but one preprocessing needs one "
                "feature space. Align the files to a common gene panel, in the same order, first."
            )
    rep = da.concatenate([_as_dask(_input(a, source), chunk_size) for a in adatas], axis=0).rechunk({1: -1})
    rows = np.flatnonzero(train_mask) if train_mask is not None else np.arange(rep.shape[0])

    steps: dict[str, dict] = {}
    if pca is not None:
        steps["pca"] = _fit_pca(rep[rows], pca, n_hvg, hvg_flavor)
        rep = _lazy(rep, "pca", steps["pca"])
    if zscore:
        mean, var = (np.asarray(v, dtype=np.float64).ravel() for v in da.compute(*stats.mean_var(rep[rows], axis=0)))
        steps["zscore"] = {"mean": mean, "std": np.sqrt(var + epsilon)}
        rep = _lazy(rep, "zscore", steps["zscore"])
    if whitening is not None:
        steps["whitening"] = _fit_whitening(rep[rows], whitening, epsilon)
        rep = _lazy(rep, "whitening", steps["whitening"])
    if not steps:
        raise ValueError("No preprocessing step is set: pass `pca`, `zscore` or `whitening`.")

    out = rep.compute()
    record = {"steps": np.array(list(steps)), "input": "X" if source is None else f"obsm/{source}", **steps}
    offsets = np.cumsum([0, *(a.n_obs for a in adatas)])
    for a, start, stop in zip(adatas, offsets[:-1], offsets[1:], strict=True):
        a.obsm[key_added] = out[start:stop]
        a.uns["preprocessing"] = record


def apply(x, record: dict) -> np.ndarray:
    """Transform ``x`` (rows of the recorded input) with the stored steps, e.g. new cells of the same features."""
    for name in record["steps"]:
        x = _forward(str(name), record[str(name)], x)
    return x


def inverse(y, record: dict) -> np.ndarray:
    """Map ``y`` (e.g. predictions in ``obsm[key_added]``'s space) back to the recorded input features.

    PCA keeps only its components, so after it this is the projection onto them.
    """
    for name in reversed(list(record["steps"])):
        y = _backward(str(name), record[str(name)], y)
    return y


def _forward(name: str, p: dict, x) -> np.ndarray:
    if name == "pca":
        return np.asarray(x @ p["PCs"]) - p["mean"] @ p["PCs"]
    x = _dense(x)
    if name == "zscore":
        return (x - p["mean"]) / p["std"]
    return (x - p["mean"]) @ p["W"]


def _backward(name: str, p: dict, y) -> np.ndarray:
    y = _dense(y)
    if name == "pca":
        return y @ p["PCs"].T + p["mean"]
    if name == "zscore":
        return y * p["std"] + p["mean"]
    return y @ p["W_inv"] + p["mean"]


def _lazy(rep: da.Array, name: str, p: dict) -> da.Array:
    width = p["PCs"].shape[1] if name == "pca" else rep.shape[1]
    return rep.map_blocks(
        lambda block: _forward(name, p, block).astype(np.float32), dtype=np.float32, chunks=(rep.chunks[0], (width,))
    )


def _fit_pca(train: da.Array, n_comps: int, n_hvg: int | None, hvg_flavor: str) -> dict:
    fit = AnnData(train)
    if n_hvg is not None:
        sc.pp.highly_variable_genes(fit, n_top_genes=n_hvg, flavor=hvg_flavor)
    sc.pp.pca(fit, n_comps, mask_var="highly_variable" if n_hvg is not None else None, svd_solver="covariance_eigh")
    summary = {k: np.asarray(da.compute(v)[0]) for k, v in fit.uns["pca"].items() if k != "params"}
    mean = np.asarray(da.compute(stats.mean(fit.X, axis=0))[0], dtype=np.float64).ravel()
    return {"PCs": fit.varm["PCs"].astype(np.float64), "mean": mean, **summary}


def _fit_whitening(train: da.Array, method: WhiteningMethod, epsilon: float) -> dict:
    """``W`` with unit covariance for ``(x - mean) @ W`` over the training rows, and its inverse."""
    dense = train.map_blocks(_dense, dtype=np.float64)
    n = dense.shape[0]
    mean, gram = da.compute(dense.mean(axis=0), dense.T @ dense)
    cov = gram / n - np.outer(mean, mean) + epsilon * np.eye(len(mean))
    if method == "cholesky":
        lower = np.linalg.cholesky(cov)
        W, W_inv = scipy.linalg.solve_triangular(lower, np.eye(len(mean)), lower=True).T, lower.T
    else:
        values, vectors = np.linalg.eigh(cov)
        if method == "pca":
            W, W_inv = vectors / np.sqrt(values), (vectors * np.sqrt(values)).T
        elif method == "zca":
            W, W_inv = (vectors / np.sqrt(values)) @ vectors.T, (vectors * np.sqrt(values)) @ vectors.T
        else:
            raise ValueError(f"Unknown whitening {method!r}; use 'zca', 'pca' or 'cholesky'.")
    return {"mean": mean, "W": W, "W_inv": W_inv, "method": method}


def _input(adata: AnnData, key: str | None):
    if key is None:
        return adata.X
    if key not in adata.obsm:
        raise KeyError(f"obsm[{key!r}] is missing; without `pca` the preprocessing starts from it.")
    return adata.obsm[key]


def _as_dask(x, chunk_size: int) -> da.Array:
    """``x`` as a dask array chunked by rows, without loading a backed one."""
    if isinstance(x, ad.abc.CSRDataset):  # backed sparse: read lazily, chunk by chunk
        return ad.experimental.read_elem_lazy(x.group, chunks=(chunk_size, x.shape[1]))
    if isinstance(x, zarr.Array):  # backed dense
        return da.from_zarr(x, chunks=(chunk_size, -1))
    if isinstance(x, da.Array):
        return x
    return da.from_array(x, chunks=(chunk_size, -1), asarray=False)


def _dense(x) -> np.ndarray:
    return np.asarray(x.toarray() if issparse(x) else x, dtype=np.float64)


def write_preprocessing(path: str | Path, adata: AnnData, key: str = "X_pca", *, overwrite: bool = False) -> None:
    """Write :func:`preprocess`'s output into the AnnData file at ``path`` (``.zarr`` or ``.h5ad``).

    Writes ``obsm[key]`` and ``uns["preprocessing"]``. Existing entries are only replaced with ``overwrite=True``,
    so a representation already in the file is never lost silently.
    """
    with open_store(path) as root:
        entries = (f"obsm/{key}", "uns/preprocessing")
        existing = [e for e in entries if e in root]
        if existing and not overwrite:
            raise ValueError(
                f"{path} already has {existing}; set overwrite=True (preprocessing.overwrite=true) to replace them."
            )
        for e in existing:
            del root[e]
        for group in ("obsm", "uns"):
            root.require_group(group)
        if isinstance(root, zarr.Group):
            from annbatch import write_sharded

            write_sharded(root["obsm"], adata.obsm[key], key=key)  # row chunks, the layout annbatch streams
        else:
            ad.io.write_elem(root["obsm"], key, adata.obsm[key])
        ad.io.write_elem(root["uns"], "preprocessing", adata.uns["preprocessing"])
