"""Edit an AnnData file in place: ``.zarr`` or ``.h5ad``."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import zarr


@contextmanager
def open_store(path: str | Path) -> Iterator[zarr.Group | object]:
    """The file's root group, writable. Zarr metadata is re-consolidated on exit, since consolidated stores refuse edits."""
    path = Path(path)
    if path.suffix == ".zarr":
        root = zarr.open_group(path, mode="r+", use_consolidated=False)
        yield root
        zarr.consolidate_metadata(path)
    else:
        import h5py

        with h5py.File(path, "r+") as root:
            yield root


def write_obs_column(path: str | Path, name: str, values: pd.Categorical | np.ndarray) -> None:
    """Add (or replace) one ``obs`` column in the file, without reading or rewriting the other columns.

    On disk ``obs`` is a group with one element per column and a ``column-order`` attribute, so only that
    element and the attribute change.
    """
    with open_store(path) as root:
        obs = root["obs"]
        order = [str(c) for c in obs.attrs["column-order"]]
        if name in obs:
            del obs[name]
        ad.io.write_elem(obs, name, values)
        if name not in order:
            obs.attrs["column-order"] = [*order, name]
