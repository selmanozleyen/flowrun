from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from anndata.acc import A
from scipy import sparse

from flowrun.io import keys, open_adata, open_inputs, read_rows, resolve


@pytest.fixture
def adata() -> ad.AnnData:
    rng = np.random.default_rng(0)
    return ad.AnnData(
        X=rng.normal(size=(6, 3)).astype(np.float32),
        obs=pd.DataFrame(
            {"group": pd.Categorical(list("aabbcc")), "value": np.arange(6.0)}, index=[f"c{i}" for i in range(6)]
        ),
        var=pd.DataFrame(index=["m0", "m1", "m2"]),
        obsm={"emb": rng.normal(size=(6, 2))},
        layers={"counts": sparse.random(6, 3, density=0.5, format="csr", random_state=0)},
        uns={"table": "kept", "big_unused": np.ones(100)},
    )


@pytest.fixture(params=[".zarr", ".h5ad"])
def store(request, adata: ad.AnnData, tmp_path: Path) -> Path:
    path = tmp_path / f"data{request.param}"
    adata.write_zarr(path) if request.param == ".zarr" else adata.write_h5ad(path)
    return path


def test_reads_only_the_referenced_parts(store: Path, adata: ad.AnnData):
    opened = open_adata(store, A.X, A.layers["counts"], A.obs["group"], uns=["table"])
    assert list(opened.obs.columns) == ["group"] and list(opened.uns) == ["table"]
    assert [k for k in opened.layers if k is not None] == ["counts"] and list(opened.obsm) == []  # None is X
    assert list(opened.var_names) == list(adata.var_names)
    assert list(opened.obs_names) == [str(i) for i in range(6)]  # no index asked for: a RangeIndex
    np.testing.assert_array_equal(opened.obs["group"], adata.obs["group"])
    np.testing.assert_allclose(read_rows(opened.X), adata.X)
    np.testing.assert_allclose(read_rows(opened.layers["counts"]), adata.layers["counts"].toarray())


def test_zarr_arrays_stay_on_disk_h5ad_arrays_are_read(store: Path):
    opened = open_adata(store, A.X, A.obsm["emb"])
    assert isinstance(opened.X, np.ndarray) == (store.suffix == ".h5ad")
    assert isinstance(opened.obsm["emb"], np.ndarray) == (store.suffix == ".h5ad")


def test_index_on_request(store: Path, adata: ad.AnnData):
    assert list(open_adata(store, A.obs["value"], A.obs.index).obs_names) == list(adata.obs_names)


def test_a_reference_into_an_array_opens_the_array(store: Path, adata: ad.AnnData):
    opened = open_adata(store, A.obsm["emb"][:, 0])
    np.testing.assert_allclose(read_rows(opened.obsm["emb"]), adata.obsm["emb"])


def test_missing_parts_name_what_the_store_has(store: Path):
    with pytest.raises(KeyError, match=r"obsm\['nope'\] \(has obsm: \['emb'\]\).*obs \['zz'\].*uns \['gone'\]"):
        open_adata(store, A.obsm["nope"], A.obs["zz"], uns=["gone"])


def test_unsupported_references_raise(store: Path):
    with pytest.raises(ValueError, match="only X, layers, obsm"):
        open_adata(store, A.var["x"])


def test_read_rows_keeps_the_requested_order(store: Path, adata: ad.AnnData):
    opened = open_adata(store, A.X, A.layers["counts"])
    np.testing.assert_allclose(read_rows(opened.X, [4, 1, 3]), adata.X[[4, 1, 3]])
    np.testing.assert_allclose(read_rows(opened.layers["counts"], [4, 1]), adata.layers["counts"][[4, 1]].toarray())
    mask = (adata.obs["group"] == "b").to_numpy()
    np.testing.assert_allclose(read_rows(opened.X, mask), adata.X[mask])


def test_config_strings_resolve_to_references():
    assert resolve("X") == A.X
    assert resolve("obsm/emb") == A.obsm["emb"] and resolve("obsm.emb") == A.obsm["emb"]
    assert resolve("layers/counts") == A.layers["counts"]


def test_keys_and_open_inputs(store: Path):
    assert keys(store, "obsm") == ["emb"] and keys(store, "uns") == ["big_unused", "table"]
    assert [a.n_obs for a in open_inputs([store, store], A.X)] == [6, 6]
