import anndata as ad
import numpy as np
import pytest
import scipy.sparse as sp
import zarr
from anndata import AnnData
from sklearn.decomposition import PCA

from flowrun.pca import pca, write_pca


def test_matches_sklearn_on_the_training_rows():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 12)) @ rng.normal(size=(12, 12))
    train = np.arange(200) < 150
    adata = AnnData(sp.csr_matrix(X))
    pca(adata, 4, train_mask=train, chunk_size=64)

    ref = PCA(4).fit(X[train])
    signs = np.sign((adata.varm["PCs"] * ref.components_.T).sum(0))
    np.testing.assert_allclose(adata.varm["PCs"] * signs, ref.components_.T, atol=1e-5)
    np.testing.assert_allclose(adata.obsm["X_pca"] * signs, ref.transform(X), atol=1e-4)
    np.testing.assert_allclose(adata.uns["pca"]["mean"], X[train].mean(0), atol=1e-5)


def test_full_rank_round_trips():
    X = np.random.default_rng(1).normal(size=(50, 6))
    adata = AnnData(X)
    pca(adata, 6)
    back = adata.obsm["X_pca"] @ adata.varm["PCs"].T + adata.uns["pca"]["mean"]
    np.testing.assert_allclose(back, X, atol=1e-4)


def test_backed_sparse_zarr_fit_on_training_rows(tmp_path):
    rng = np.random.default_rng(2)
    X = rng.normal(size=(120, 8)).astype(np.float32)
    train = np.arange(120) % 3 != 0
    AnnData(sp.csr_matrix(X)).write_zarr(tmp_path / "a.zarr")
    adata = AnnData(ad.io.sparse_dataset(zarr.open_group(tmp_path / "a.zarr")["X"]))
    pca(adata, 3, train_mask=train, chunk_size=16)

    ref = PCA(3).fit(X[train])
    np.testing.assert_allclose(np.abs(adata.varm["PCs"]), np.abs(ref.components_.T), atol=1e-4)
    np.testing.assert_allclose(np.abs(adata.obsm["X_pca"]), np.abs(ref.transform(X)), atol=1e-4)


@pytest.mark.parametrize("suffix", [".zarr", ".h5ad"])
def test_write_pca_into_the_input_file(tmp_path, suffix):
    X = np.random.default_rng(3).normal(size=(40, 6)).astype(np.float32)
    path = tmp_path / f"a{suffix}"
    write = AnnData.write_zarr if suffix == ".zarr" else AnnData.write_h5ad
    write(AnnData(X), path)
    adata = AnnData(X)
    pca(adata, 3)
    write_pca(path, adata)

    back = (ad.read_zarr if suffix == ".zarr" else ad.read_h5ad)(path)
    np.testing.assert_allclose(back.obsm["X_pca"], adata.obsm["X_pca"])
    np.testing.assert_allclose(back.varm["PCs"], adata.varm["PCs"])
    np.testing.assert_allclose(back.uns["pca"]["mean"], adata.uns["pca"]["mean"])
    with pytest.raises(ValueError, match="already has"):
        write_pca(path, adata)
    write_pca(path, adata, overwrite=True)
