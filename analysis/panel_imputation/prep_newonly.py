"""Slim copies of the two prepared panels for the new-only flow: the shared markers recalibrated, the new-only target.

    python prep_newonly.py <prepared dir> <out dir> [split key]

obsm, per file:
- ``shared_recal``: the 16 shared markers (logicle), the old panel's quantile-matched to the new panel's train cells
  (per marker, F_new^-1(F_old(x))); the new panel's as measured. The flow's source state.
- ``lin``: the same, the fused GW coupling's shared (linear) term.
- ``quad_old`` / ``quad_new``: each panel's full marker set, the coupling's within-panel (quadratic) term.
- ``new_only`` (new panel): the markers only the new panel measures. The flow's target state.
"""

import json
import sys
from pathlib import Path

import anndata as ad
import numpy as np


def quantile_match(x: np.ndarray, reference: np.ndarray, n_q: int = 1001) -> np.ndarray:
    q = np.linspace(0, 1, n_q)
    out = np.empty_like(x)
    for j in range(x.shape[1]):
        out[:, j] = np.interp(x[:, j], np.quantile(x[:, j], q), np.quantile(reference[:, j], q))
    return out


def main() -> None:
    src, out = Path(sys.argv[1]), Path(sys.argv[2])
    split = sys.argv[3] if len(sys.argv) > 3 else "split_42"
    out.mkdir(parents=True, exist_ok=True)
    old = ad.read_zarr(src / "OH_OH0013_OLD_UMAP_annotated_red.zarr")
    new = ad.read_zarr(src / "260331_OH013_New_Panel_annotated_red.zarr")
    lo, ln = (np.asarray(a.layers["logicle_trans"], dtype=np.float32) for a in (old, new))
    shared = [m for m in new.var_names if m in set(old.var_names)]
    new_only = [m for m in new.var_names if m not in set(old.var_names)]
    o_idx = [list(old.var_names).index(m) for m in shared]
    n_idx = [list(new.var_names).index(m) for m in shared]
    train = (new.obs[split].astype(str) == "train").to_numpy()
    print(f"old {old.n_obs} x {old.n_vars}, new {new.n_obs} x {new.n_vars}; {len(shared)} shared, "
          f"{len(new_only)} new-only, {old.n_vars - len(shared)} old-only; new train {train.sum()}", flush=True)

    old_shared = quantile_match(lo[:, o_idx], ln[train][:, n_idx]).astype(np.float32)
    new_shared = ln[:, n_idx]
    for m, a, b in zip(shared, lo[:, o_idx].T, old_shared.T, strict=True):
        print(f"  {m:8s} old median {np.median(a):.2f} -> recalibrated {np.median(b):.2f} "
              f"(new median {np.median(ln[train][:, n_idx[shared.index(m)]]):.2f})")

    keep_obs = [c for c in ("dataset", split) if c in old.obs]
    for a, x, rep, quad_key in ((old, lo, old_shared, "quad_old"), (new, ln, new_shared, "quad_new")):
        slim = ad.AnnData(X=x, obs=a.obs[[c for c in keep_obs if c in a.obs]].copy(), var=a.var[[]].copy())
        slim.layers["logicle_trans"] = x
        slim.obsm["shared_recal"] = rep
        slim.obsm["lin"] = rep
        slim.obsm[quad_key] = x
        if a is new:
            slim.obsm["new_only"] = x[:, [list(new.var_names).index(m) for m in new_only]]
        slim.uns = {k: a.uns[k] for k in ("dataset",) if k in a.uns}
        name = "old_panel.zarr" if a is old else "new_panel.zarr"
        slim.write_zarr(out / name)
        print(f"wrote {out / name}: obsm {list(slim.obsm)}, obs {list(slim.obs.columns)}, uns {list(slim.uns)}",
              flush=True)
    (out / "markers.json").write_text(json.dumps({"shared": shared, "new_only": new_only}, indent=1))


if __name__ == "__main__":
    main()
