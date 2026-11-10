"""The panels' obs annotations, and the GMM's K chosen by per-cell R^2 on the validation cells.

    python obs_and_k.py <prepared dir> <out dir> [split key]
"""

import json
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import zarr
from sklearn.metrics import r2_score
from sklearn.mixture import GaussianMixture

sys.path.insert(0, str(Path(__file__).parent))
from gmm_percell import gmm_conditional  # noqa: E402

FILES = {"old": "OH_OH0013_OLD_UMAP_annotated_red.zarr", "new": "260331_OH013_New_Panel_annotated_red.zarr"}


def main() -> None:
    src, out = Path(sys.argv[1]), Path(sys.argv[2])
    split = sys.argv[3] if len(sys.argv) > 3 else "split_42"
    out.mkdir(parents=True, exist_ok=True)

    # 1. obs annotations: every column, value counts of the categorical-looking ones
    obs = {}
    for name, f in FILES.items():
        o = ad.io.read_elem(zarr.open_group(src / f, mode="r")["obs"])
        obs[name] = o
        print(f"\n===== {name} panel: {len(o)} cells, obs columns: {list(o.columns)}", flush=True)
        for c in o.columns:
            n = o[c].nunique()
            if 2 <= n <= 100 and not c.startswith("split_"):
                vc = o[c].value_counts()
                print(f"--- {c} ({n} values)", flush=True)
                print("    " + ", ".join(f"{k}: {v}" for k, v in vc.items()), flush=True)

    # 2. K by per-cell R^2 on the new panel's validation cells
    new = ad.read_zarr(src / FILES["new"])
    old_vars = set(ad.read_zarr(src / FILES["old"]).var_names)
    x = np.asarray(new.layers["logicle_trans"], dtype=np.float64)
    markers = list(new.var_names)
    s_idx = [j for j, m in enumerate(markers) if m in old_vars]
    n_idx = [j for j, m in enumerate(markers) if m not in old_vars]
    labels = new.obs[split].astype(str).to_numpy()
    train, val = x[labels == "train"], x[labels == "val"]
    rng = np.random.default_rng(0)
    fit = train[rng.choice(len(train), size=min(60000, len(train)), replace=False)]
    print(f"\ntrain {len(train)} (fit on {len(fit)}), val {len(val)}", flush=True)
    rows = []
    for K in (8, 12, 16, 20, 24, 28, 32, 40):
        for seed in (0, 1):
            gmm = GaussianMixture(K, covariance_type="full", reg_covar=1e-4, random_state=seed, max_iter=300).fit(fit)
            mean, _, resp = gmm_conditional(gmm, val[:, s_idx], s_idx, n_idx, np.random.default_rng(seed))
            r2 = [r2_score(val[:, j], mean[:, i]) for i, j in enumerate(n_idx)]
            rows.append({"K": K, "seed": seed, "BIC": gmm.bic(fit), "val R2 mean": float(np.mean(r2)),
                         "mean top responsibility": resp})
            print(json.dumps(rows[-1]), flush=True)
    table = pd.DataFrame(rows).groupby("K")[["BIC", "val R2 mean", "mean top responsibility"]].mean()
    table.to_csv(out / "k_sweep.csv")
    print(table.round(4).to_string(), flush=True)


if __name__ == "__main__":
    main()
