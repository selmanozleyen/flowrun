"""PCA fitted on the training cells, written into the AnnData.

Based on cf-train's ``myapp-pca``: optionally pick HVGs (scanpy), accumulate the gene covariance
``XᵀX`` over the training rows in chunks, eigendecompose it, then project every cell in chunks.
Only the training rows are read for the fit, chunk by chunk, so a backed AnnData
(``anndata.read_h5ad(path, backed="r")``) never loads held-out cells. The fitted basis is stored in the
AnnData, so predictions map back to genes with ``X_pca @ adata.varm["PCs"].T + adata.var["pca_mean"]``.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from anndata import AnnData
from scipy.sparse import issparse


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
        Cells by genes; ``X`` may be dense, sparse or backed.
    n_comps
        Number of components.
    train_mask
        Boolean mask over the rows the basis (and the HVGs) are fitted on. ``None`` fits on every row.
        Pass the training split, so held-out cells never shape the space they are scored in.
    n_hvg
        Restrict the fit to this many highly variable genes; ``None`` uses every gene.
    hvg_flavor
        Flavor for :func:`scanpy.pp.highly_variable_genes`. ``"seurat"`` expects log-normalised data;
        use ``"seurat_v3"`` for raw counts (needs ``scikit-misc``).
    key_added
        Key of the projection in ``obsm``.
    chunk_size
        Rows per chunk when accumulating the covariance and projecting.

    Writes ``obsm[key_added]``, ``varm["PCs"]`` (zero for genes outside the fit), ``var["pca_mean"]``
    and ``uns["pca"]["variance"]``.
    """
    rows = np.flatnonzero(train_mask) if train_mask is not None else np.arange(adata.n_obs)
    genes = np.arange(adata.n_vars)
    if n_hvg is not None:
        import scanpy as sc

        # ponytail: HVGs see every training row at once; subsample like cf-train if that stops fitting in memory
        train = AnnData(_stack(adata.X[rows[i : i + chunk_size]] for i in range(0, rows.size, chunk_size)))
        hvg = sc.pp.highly_variable_genes(train, n_top_genes=n_hvg, flavor=hvg_flavor, inplace=False)
        genes = np.flatnonzero(hvg["highly_variable"].to_numpy())

    gram = np.zeros((genes.size, genes.size))
    total = np.zeros(genes.size)
    for i in range(0, rows.size, chunk_size):
        X = _dense(adata.X[rows[i : i + chunk_size]][:, genes])
        gram += X.T @ X
        total += X.sum(0)
    mean = total / rows.size
    variance, vectors = np.linalg.eigh(gram / rows.size - np.outer(mean, mean))  # ascending
    pcs = vectors[:, ::-1][:, :n_comps]

    out = np.empty((adata.n_obs, pcs.shape[1]), dtype=np.float32)
    shift = mean @ pcs
    for i in range(0, adata.n_obs, chunk_size):
        out[i : i + chunk_size] = _dense(adata.X[i : i + chunk_size][:, genes]) @ pcs - shift

    adata.obsm[key_added] = out
    adata.varm["PCs"] = np.zeros((adata.n_vars, pcs.shape[1]), dtype=np.float32)
    adata.varm["PCs"][genes] = pcs
    full_mean = np.zeros(adata.n_vars, dtype=np.float32)
    full_mean[genes] = mean
    adata.var["pca_mean"] = full_mean
    adata.uns["pca"] = {"variance": variance[::-1][:n_comps].astype(np.float32)}


def _stack(chunks):
    chunks = list(chunks)
    return sp.vstack(chunks, format="csr") if issparse(chunks[0]) else np.vstack(chunks)


def _dense(X) -> np.ndarray:
    return np.asarray(X.toarray() if issparse(X) else X, dtype=np.float64)
