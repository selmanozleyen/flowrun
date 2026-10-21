import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
import zarr
from anndata import AnnData
from sklearn.decomposition import PCA

from flowrun._store import write_obs_column
from flowrun.preprocess import apply, inverse, preprocess, write_preprocessing


def _data(n=120, d=8, seed=0):
    rng = np.random.default_rng(seed)
    return (rng.normal(size=(n, d)) @ rng.normal(size=(d, d)) + rng.normal(size=d)).astype(np.float32)


def test_pca_matches_sklearn_on_the_training_rows():
    X = _data()
    train = np.arange(len(X)) < 90
    adata = AnnData(sp.csr_matrix(X))
    preprocess(adata, train_mask=train, pca=4, chunk_size=32)

    ref = PCA(4).fit(X[train])
    pcs = adata.uns["preprocessing"]["pca"]["PCs"]
    signs = np.sign((pcs * ref.components_.T).sum(0))  # a component's sign is arbitrary
    np.testing.assert_allclose(pcs * signs, ref.components_.T, atol=1e-5)
    np.testing.assert_allclose(adata.obsm["X_pca"] * signs, ref.transform(X), atol=1e-4)


@pytest.mark.parametrize("method", ["zca", "pca", "cholesky"])
def test_whitening_gives_unit_covariance_on_the_training_rows_and_inverts(method):
    X = _data()
    train = np.arange(len(X)) < 90
    adata = AnnData(X)
    preprocess(adata, train_mask=train, whitening=method)
    Y = adata.obsm["X_pca"]
    np.testing.assert_allclose(np.cov(Y[train].T, bias=True), np.eye(X.shape[1]), atol=1e-3)
    np.testing.assert_allclose(inverse(Y, adata.uns["preprocessing"]), X, atol=1e-3)


def test_zscore_standardises_on_the_training_rows_and_inverts():
    X = _data()
    train = np.arange(len(X)) < 90
    adata = AnnData(X)
    preprocess(adata, train_mask=train, zscore=True)
    Y = adata.obsm["X_pca"]
    np.testing.assert_allclose(Y[train].mean(0), 0, atol=1e-4)
    np.testing.assert_allclose(Y[train].std(0), 1, atol=1e-3)
    np.testing.assert_allclose(inverse(Y, adata.uns["preprocessing"]), X, atol=1e-4)


def test_the_chain_is_stored_and_repeats_on_new_cells():
    X = _data(d=10)
    adata = AnnData(X)
    preprocess(adata, pca=10, zscore=True, whitening="zca")
    record = adata.uns["preprocessing"]
    assert list(record["steps"]) == ["pca", "zscore", "whitening"] and record["input"] == "X"
    np.testing.assert_allclose(apply(X[:5], record), adata.obsm["X_pca"][:5], atol=1e-4)  # same transformation
    np.testing.assert_allclose(inverse(adata.obsm["X_pca"], record), X, atol=1e-3)  # full rank: back to genes


def test_without_pca_it_starts_from_an_existing_rep():
    X = _data()
    adata = AnnData(np.zeros((len(X), 3)), obsm={"X_emb": X})
    preprocess(adata, zscore=True, input_key="X_emb", key_added="X_emb_z")
    assert adata.uns["preprocessing"]["input"] == "obsm/X_emb"
    np.testing.assert_allclose(adata.obsm["X_emb_z"].mean(0), 0, atol=1e-4)


def test_a_list_is_one_dataset():
    X = _data(n=60)
    train = np.random.default_rng(4).random(60) < 0.7
    whole = AnnData(X)
    preprocess(whole, train_mask=train, pca=3, zscore=True)
    parts = [AnnData(X[:25]), AnnData(X[25:])]
    preprocess(parts, train_mask=train, pca=3, zscore=True)
    signs = np.sign((parts[0].uns["preprocessing"]["pca"]["PCs"] * whole.uns["preprocessing"]["pca"]["PCs"]).sum(0))
    np.testing.assert_allclose(np.vstack([p.obsm["X_pca"] for p in parts]) * signs, whole.obsm["X_pca"], atol=1e-3)


def test_a_list_needs_one_gene_space():
    a, b = AnnData(np.ones((4, 3))), AnnData(np.ones((4, 3)))
    b.var_names = ["x", "y", "z"]
    with pytest.raises(ValueError, match="AnnData 1 has different genes"):
        preprocess([a, b], pca=2)


def test_backed_sparse_zarr_reads_lazily(tmp_path):
    X = _data()
    train = np.arange(len(X)) % 3 != 0
    AnnData(sp.csr_matrix(X)).write_zarr(tmp_path / "a.zarr")
    adata = AnnData(ad.io.sparse_dataset(zarr.open_group(tmp_path / "a.zarr")["X"]))
    preprocess(adata, train_mask=train, pca=3, chunk_size=16)
    ref = PCA(3).fit(X[train])
    np.testing.assert_allclose(np.abs(adata.obsm["X_pca"]), np.abs(ref.transform(X)), atol=1e-4)


@pytest.mark.parametrize("suffix", [".zarr", ".h5ad"])
def test_the_preprocessing_is_written_into_the_input_file(tmp_path, suffix):
    X = _data()
    path = tmp_path / f"a{suffix}"
    (AnnData.write_zarr if suffix == ".zarr" else AnnData.write_h5ad)(AnnData(X), path)
    adata = AnnData(X)
    preprocess(adata, pca=4, whitening="pca")
    write_preprocessing(path, adata)

    back = (ad.read_zarr if suffix == ".zarr" else ad.read_h5ad)(path)
    np.testing.assert_allclose(back.obsm["X_pca"], adata.obsm["X_pca"])
    np.testing.assert_allclose(apply(X, back.uns["preprocessing"]), adata.obsm["X_pca"], atol=1e-4)
    with pytest.raises(ValueError, match="already has"):
        write_preprocessing(path, adata)
    write_preprocessing(path, adata, overwrite=True)


@pytest.mark.parametrize("suffix", [".zarr", ".h5ad"])
def test_write_obs_column_keeps_the_rest(tmp_path, suffix):
    path = tmp_path / f"a{suffix}"
    adata = AnnData(np.zeros((4, 2)), obs={"cell_line": ["A", "A", "B", "B"]})
    (adata.write_zarr if suffix == ".zarr" else adata.write_h5ad)(path)
    write_obs_column(path, "split", pd.Categorical(["train", "test", "all", "train"]))
    write_obs_column(path, "split", pd.Categorical(["train", "val", "all", "train"]))  # replaces, keeps order

    back = (ad.read_zarr if suffix == ".zarr" else ad.read_h5ad)(path)
    assert list(back.obs.columns) == ["cell_line", "split"]
    assert back.obs["split"].tolist() == ["train", "val", "all", "train"]
    assert back.obs["cell_line"].tolist() == ["A", "A", "B", "B"]
