"""Cells measured on different, named feature spaces (e.g. marker panels), one file per space.

:func:`panel_reps` runs in ``fit_cfm`` (``panels.layer``): from one layer, as is, each file gets in memory

- ``obsm["X_state"]``: all of its features, the state;
- ``obsm["X_lin"]``: the features every file measures, by name, sorted, the coupling's shared term;
- ``obsm["X_quad"]``: its own features, sorted, the coupling's quadratic term.

Synonyms must be renamed and the files labelled beforehand, by the project (e.g. theislab/flow-cytometry-imputation).
"""

from __future__ import annotations

from collections.abc import Sequence

import anndata as ad
import numpy as np


def panel_reps(adatas: Sequence[ad.AnnData], *, layer: str) -> dict:
    """Adds each file's state, shared and own features from ``layers[layer]``, in place (see the module)."""
    names = [list(a.var_names) for a in adatas]
    shared = sorted(set(names[0]).intersection(*names[1:]))
    if not shared:
        raise ValueError("The files share no feature name; rename synonyms when preparing them.")
    own = []
    for a, n in zip(adatas, names, strict=True):
        values = np.asarray(a.layers[layer][:], dtype=np.float32)
        col = {v: i for i, v in enumerate(n)}
        own.append(sorted(set(n) - set(shared)))
        a.obsm["X_state"] = values
        a.obsm["X_lin"] = values[:, [col[v] for v in shared]]
        a.obsm["X_quad"] = values[:, [col[v] for v in own[-1]]]
    return {"layer": layer, "shared": shared, "own": own}
