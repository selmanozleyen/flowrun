"""Split labels for ``adata.obs``: each function reads only ``obs`` and returns one label per row.

Labels are ``"train"``, ``"val"``, ``"test"`` and ``"control"``. Controls are the shared source population,
so they are never split. Write the result to ``adata.obs[split_by]``; sckitflow's ``DataManagerConfig(split_by=...)``
reads it from there.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence

import numpy as np
import pandas as pd


def holdout(
    obs: pd.DataFrame,
    key: str,
    *,
    test: Sequence[str] = (),
    val: Sequence[str] = (),
    control_key: str | None = None,
    control_value: str = "control",
) -> pd.Series:
    """Hold out chosen values of one categorical, e.g. whole cell lines.

    Rows whose ``obs[key]`` is in ``test`` (``val``) are labelled ``"test"`` (``"val"``), the rest ``"train"``.
    """
    values = obs[key].astype(str)
    unknown = set(map(str, (*test, *val))) - set(values)
    if unknown:
        raise ValueError(f"{sorted(unknown)} not found in obs[{key!r}]; it has {sorted(set(values))}.")
    labels = np.full(len(obs), "train", dtype=object)
    labels[values.isin(list(map(str, val))).to_numpy()] = "val"
    labels[values.isin(list(map(str, test))).to_numpy()] = "test"
    return _with_controls(obs, labels, control_key, control_value)


def combination(
    obs: pd.DataFrame,
    group_keys: Sequence[str],
    *,
    always_train_keys: Sequence[str] = (),
    control_key: str | None = None,
    control_value: str = "control",
    test_fraction: float = 0.2,
    seed: int = 0,
) -> pd.Series:
    """Hold out whole ``group_keys`` combinations, so a held-out combination is unseen in training.

    Combinations are grouped by their ``always_train_keys`` values (a stratum), and a stratum of ``k``
    combinations gives ``floor(test_fraction * k)`` of them to test, never more than ``k - 1``. With
    ``group_keys=["cell_line", "drug"]`` and ``always_train_keys=["cell_line"]``, every cell line keeps some
    drugs in train.
    """
    if not set(always_train_keys) <= set(group_keys):
        raise ValueError(
            f"always_train_keys {list(always_train_keys)} must be a subset of group_keys {list(group_keys)}."
        )
    is_control = _is_control(obs, control_key, control_value)
    gk, atk = list(group_keys), list(always_train_keys)
    combos = obs.loc[~is_control, gk].astype(str).drop_duplicates()

    rng = np.random.default_rng(seed)
    test_combos: list[tuple] = []
    strata = combos.groupby(atk, sort=True) if atk else [(None, combos)]
    for _, grp in strata:
        rows = list(map(tuple, grp[gk].to_numpy()))
        n_test = min(int(np.floor(test_fraction * len(rows))), len(rows) - 1)
        if n_test > 0:
            test_combos.extend(rows[i] for i in np.sort(rng.choice(len(rows), size=n_test, replace=False)))
    if test_fraction > 0 and not test_combos:
        warnings.warn(
            f"Nothing was held out: at test_fraction={test_fraction} a stratum needs at least "
            f"{int(np.ceil(1 / test_fraction))} combinations to give one to test.",
            UserWarning,
            stacklevel=2,
        )

    labels = np.full(len(obs), "train", dtype=object)
    if test_combos:
        combo_index = pd.MultiIndex.from_arrays([obs[c].astype(str).to_numpy() for c in gk])
        labels[combo_index.isin(test_combos)] = "test"
    return _with_controls(obs, labels, control_key, control_value)


def _is_control(obs: pd.DataFrame, control_key: str | None, control_value: str) -> np.ndarray:
    if control_key is None:
        return np.zeros(len(obs), dtype=bool)
    return obs[control_key].astype(str).to_numpy() == str(control_value)


def _with_controls(obs: pd.DataFrame, labels: np.ndarray, control_key: str | None, control_value: str) -> pd.Series:
    labels[_is_control(obs, control_key, control_value)] = "control"
    return pd.Series(pd.Categorical(labels), index=obs.index)
