"""Split labels for ``adata.obs``: each function reads only ``obs`` and returns one label per row.

Targets get ``"train"``, ``"val"`` or ``"test"``, so a target is in exactly one split. Sources get ``"all"``: both
splitters share them, and sckitflow draws every split's sources from all of them (matched by group). Write the
result to ``adata.obs[split_by]``; sckitflow's ``DataManagerConfig(split_by=...)`` reads it.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd


def shared_source_holdout(
    obs: pd.DataFrame,
    key: str,
    *,
    is_source: np.ndarray,
    test: Sequence[str] = (),
    val: Sequence[str] = (),
) -> pd.Series:
    """Hold out chosen values of one column, e.g. whole cell lines.

    Targets whose ``obs[key]`` is in ``test`` (``val``) are labelled ``"test"`` (``"val"``), the rest ``"train"``;
    sources ``"all"``.
    """
    values = obs[key].astype(str)
    unknown = set(map(str, (*test, *val))) - set(values)
    if unknown:
        raise ValueError(f"{sorted(unknown)} not found in obs[{key!r}]; it has {sorted(set(values))}.")
    labels = np.full(len(obs), "train", dtype=object)
    labels[values.isin(list(map(str, val))).to_numpy() & ~is_source] = "val"
    labels[values.isin(list(map(str, test))).to_numpy() & ~is_source] = "test"
    labels[is_source] = "all"
    return pd.Series(pd.Categorical(labels), index=obs.index)


def shared_source_combination(
    obs: pd.DataFrame,
    group_keys: Sequence[str],
    *,
    is_source: np.ndarray,
    always_train_keys: Sequence[str] = (),
    test_fraction: float = 0.2,
    val_fraction: float = 0.0,
    seed: int = 0,
) -> pd.Series:
    """Hold out whole ``group_keys`` combinations, so a held-out combination is unseen in training.

    Target combinations are grouped by their ``always_train_keys`` values (a stratum). A stratum of ``k``
    combinations gives ``floor(test_fraction * k)`` to test and ``floor(val_fraction * k)`` to val, keeping at
    least one in train. With ``group_keys=["cell_line", "drug"]`` and ``always_train_keys=["cell_line"]``,
    every cell line keeps some drugs in train.
    """
    if not set(always_train_keys) <= set(group_keys):
        raise ValueError(
            f"always_train_keys {list(always_train_keys)} must be a subset of group_keys {list(group_keys)}."
        )
    gk, atk = list(group_keys), list(always_train_keys)
    combos = obs.loc[~is_source, gk].astype(str).drop_duplicates()

    rng = np.random.default_rng(seed)
    held: dict[tuple, str] = {}
    strata = combos.groupby(atk, sort=True) if atk else [(None, combos)]
    for _, grp in strata:
        rows = list(map(tuple, grp[gk].to_numpy()))
        n_test = min(int(np.floor(test_fraction * len(rows))), len(rows) - 1)
        n_val = min(int(np.floor(val_fraction * len(rows))), len(rows) - 1 - n_test)
        order = rng.permutation(len(rows))
        held |= {rows[i]: "test" for i in order[:n_test]} | {rows[i]: "val" for i in order[n_test : n_test + n_val]}
    if (test_fraction > 0 or val_fraction > 0) and not held:
        smallest = int(np.ceil(1 / max(test_fraction, val_fraction)))
        raise ValueError(
            f"Nothing was held out: at test_fraction={test_fraction}, val_fraction={val_fraction} a stratum needs "
            f"at least {smallest} combinations to give one away. Use more group_keys combinations, or prepare a "
            "split column in the input file."
        )

    keys = pd.MultiIndex.from_arrays([obs[c].astype(str).to_numpy() for c in gk])
    labels = np.asarray([held.get(k, "train") for k in keys], dtype=object)
    labels[is_source] = "all"
    return pd.Series(pd.Categorical(labels), index=obs.index)
