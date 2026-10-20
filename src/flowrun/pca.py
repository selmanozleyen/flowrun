"""PCA fitted on the training cells with :func:`scanpy.pp.pca` on dask, written into the AnnData.

``X`` is handled as a dask array chunked by rows (a backed zarr ``X`` is wrapped without loading it), so the fit
reads only the training rows' chunks: HVGs and the basis come from those rows, then every row is projected into that basis.
Predictions map back to genes with ``X_pca @ adata.varm["PCs"].T + adata.uns["pca"]["mean"]``. :func:`write_pca`
stores the result in the AnnData file it came from.

TODO(scale): the cost grows with genes, not cells -- scanpy's dask PCA (``covariance_eigh``) builds a
genes x genes covariance, so ``n_hvg=None`` on ~20k genes needs ~3 GB and minutes of ``eigh``. For larger data:
default to HVGs (as cf-train does), and add a GPU path (rapids-singlecell, or cf-train's cupy loop).
"""

from __future__ import annotations

from pathlib import Path

import anndata as ad
import dask.array as da
import numpy as np
import scanpy as sc
import zarr
from anndata import AnnData
from fast_array_utils import stats


def pca(
    adata: AnnData,
    n_comps: int,
    *,
    train_mask: np.ndarray | None = None,
    n_hvg: int | None = None,
    hvg_flavor: str = "seurat",
    key_added: str = "X_pca",
    chunk_size: int = 10_000,
) -> None:
    """Fit a PCA of ``adata.X`` on the training rows and project every row, in place.

    Parameters
    ----------
    adata
        Cells by genes; ``X`` may be in memory, backed on disk (zarr), or dask.
    n_comps
        Number of components.
    train_mask
        Boolean mask over the rows the basis (and the HVGs) are fitted on. ``None`` fits on every row.
        Pass the training split, so held-out cells never shape the space they are scored in.
    n_hvg
        Restrict the fit to this many highly variable genes; ``None`` uses every gene.
    hvg_flavor
        Flavor for :func:`scanpy.pp.highly_variable_genes`. ``"seurat"`` expects log-normalised data.
    key_added
        Key of the projection in ``obsm``.
    chunk_size
        Rows per dask chunk when ``X`` is not dask already.

    Writes ``obsm[key_added]`` for every row, ``varm["PCs"]`` (zero for genes outside the fit) and
    ``uns["pca"]`` (``mean``, ``variance``, ``variance_ratio``).
    """
    X = adata.X
    if isinstance(X, ad.abc.CSRDataset):  # backed sparse: read lazily, chunk by chunk
        X = ad.experimental.read_elem_lazy(X.group, chunks=(chunk_size, adata.n_vars))
    elif isinstance(X, zarr.Array):  # backed dense
        X = da.from_zarr(X, chunks=(chunk_size, -1))
    elif not isinstance(X, da.Array):
        X = da.from_array(X, chunks=(chunk_size, -1), asarray=False)
    X = X.rechunk({1: -1})  # scanpy's dask PCA wants whole rows per chunk
    rows = np.flatnonzero(train_mask) if train_mask is not None else np.arange(adata.n_obs)

    fit = AnnData(X[rows])
    if n_hvg is not None:
        sc.pp.highly_variable_genes(fit, n_top_genes=n_hvg, flavor=hvg_flavor)
    sc.pp.pca(fit, n_comps, mask_var="highly_variable" if n_hvg is not None else None, svd_solver="covariance_eigh")
    pcs = fit.varm["PCs"].astype(np.float32)
    mean = np.asarray(da.compute(stats.mean(fit.X, axis=0))[0], dtype=np.float32).ravel()

    shift = mean @ pcs
    projected = X.map_blocks(
        lambda block: np.asarray(block @ pcs, dtype=np.float32) - shift,
        dtype=np.float32,
        chunks=(X.chunks[0], (n_comps,)),
    )
    adata.obsm[key_added] = projected.compute()
    adata.varm["PCs"] = pcs
    adata.uns["pca"] = {k: np.asarray(da.compute(v)[0]) for k, v in fit.uns["pca"].items() if k != "params"}
    adata.uns["pca"]["mean"] = mean


def write_pca(path: str | Path, adata: AnnData, key: str = "X_pca", *, overwrite: bool = False) -> None:
    """Write :func:`pca`'s output into the AnnData file at ``path`` (``.zarr`` or ``.h5ad``).

    Writes ``obsm[key]``, ``varm["PCs"]`` and ``uns["pca"]``. Existing entries are only replaced with
    ``overwrite=True``, so a PCA already in the file is never lost silently.
    """
    path = Path(path)
    if path.suffix == ".zarr":
        from annbatch import write_sharded

        root = zarr.open_group(path, mode="r+", use_consolidated=False)  # consolidated stores refuse edits
    else:
        import h5py

        root = h5py.File(path, "r+")
    entries = (f"obsm/{key}", "varm/PCs", "uns/pca")
    existing = [e for e in entries if e in root]
    if existing and not overwrite:
        raise ValueError(f"{path} already has {existing}; set overwrite=True (pca.overwrite=true) to replace them.")
    for e in existing:
        del root[e]
    for group in ("obsm", "varm", "uns"):
        root.require_group(group)
    if path.suffix == ".zarr":
        write_sharded(root["obsm"], adata.obsm[key], key=key)  # row chunks, the layout annbatch streams
    else:
        ad.io.write_elem(root["obsm"], key, adata.obsm[key])
    ad.io.write_elem(root["varm"], "PCs", adata.varm["PCs"])
    ad.io.write_elem(root["uns"], "pca", adata.uns["pca"])
    if isinstance(root, zarr.Group):
        zarr.consolidate_metadata(path)
    else:
        root.close()
