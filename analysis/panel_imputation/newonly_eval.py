"""Evaluate the new-only flow (source: 16 shared markers in new-panel units; target: the 12 new-only markers).

    python newonly_eval.py <run dir (.hydra/, save/)> <out dir> [gmm per-cell csv]

1. Per cell, inside the new panel: the new-panel test cells' shared markers in, their hidden new-only markers out.
   R^2 of the mean over 100 samples, Pearson r and KS of one sample, per marker (next to the GMM's when given).
2. The old panel: 20k old cells (as in diagnose.py) imputed; a full cell is its recalibrated shared markers plus the
   generated new-only ones, scored on all 28 markers against the new panel's test cells (KS, energy distance, plot).
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf
from scipy.stats import ks_2samp, pearsonr
from sklearn.metrics import r2_score

sys.path.insert(0, str(Path(__file__).parent))
from diagnose import energy_distance, plot, predict  # noqa: E402
from flowrun.fit_cfm import as_list, enable_zarrs, read_zarr_input  # noqa: E402
from flowrun.panels import from_state  # noqa: E402
from sckitflow import Model  # noqa: E402
from sckitflow.data import FlowDataModule  # noqa: E402


def main() -> None:
    run, out = Path(sys.argv[1]), Path(sys.argv[2])
    gmm_csv = Path(sys.argv[3]) if len(sys.argv) > 3 else None
    out.mkdir(parents=True, exist_ok=True)
    enable_zarrs()
    cfg = OmegaConf.load(run / ".hydra" / "config.yaml")
    paths = [Path(p) for p in as_list(cfg.paths.target_adata)]
    adatas = [read_zarr_input(p, cfg) for p in paths]
    from_state(adatas, OmegaConf.to_container(cfg.datamanager, resolve=True))
    model = Model.from_pretrained(run / "save", adatas)
    model.module.to("cuda").eval()
    datamodule = FlowDataModule.from_adata(adatas, model.manager, n_train_steps=1, batch_size=256)
    step_data, _ = next(iter(datamodule.dm.get_eval_loader(adatas, max_per_group=1)))

    names = json.loads((paths[0].parent / "markers.json").read_text())
    shared, new_only = names["shared"], names["new_only"]
    is_source = [str(a.obs["dataset"].iloc[0]) == str(cfg.datamanager.control_values_dict.panel) for a in adatas]
    old, new = adatas[is_source.index(True)], adatas[is_source.index(False)]
    markers = list(new.var_names)
    s_pos, n_pos = [markers.index(m) for m in shared], [markers.index(m) for m in new_only]
    labels = new.obs[f"split_{cfg.run.splitter_seed}"].astype(str).to_numpy()
    new_full = np.asarray(new.obsm["quad_new"][:], dtype=np.float32)
    new_shared = np.asarray(new.obsm["shared_recal"][:], dtype=np.float32)
    test = labels == "test"

    # 1. per cell inside the new panel
    s_test, n_test = new_shared[test], new_full[test][:, n_pos]
    mean = predict(model, step_data, s_test, 10, 100)
    sample = predict(model, step_data, s_test, 10, 1)
    rows = []
    for i, m in enumerate(new_only):
        rows.append({"marker": m, "R2 flow mean": r2_score(n_test[:, i], mean[:, i]),
                     "r flow sample": pearsonr(n_test[:, i], sample[:, i]).statistic,
                     "KS flow sample": ks_2samp(n_test[:, i], sample[:, i]).statistic})
    table = pd.DataFrame(rows).set_index("marker")
    if gmm_csv is not None and gmm_csv.exists():
        gmm = pd.read_csv(gmm_csv, index_col=0)
        table = table.join(gmm[["R2 gmm mean", "r gmm sample", "KS gmm sample"]], how="left")
    table = table.sort_values("R2 flow mean")
    table.loc["mean"] = table.mean()
    table.to_csv(out / "percell_new_only.csv")
    print(table.round(3).to_string(), flush=True)

    # 2. the old panel: recalibrated shared markers kept, new-only generated
    real = new_full[test]
    old_shared = np.asarray(old.obsm["shared_recal"][:], dtype=np.float32)
    old_full = np.asarray(old.obsm["quad_old"][:], dtype=np.float32)
    pick = np.sort(np.random.default_rng(0).choice(old.n_obs, size=min(20000, old.n_obs), replace=False))
    gen = np.empty((len(pick), len(markers)), dtype=np.float32)
    gen[:, s_pos] = old_shared[pick]
    gen[:, n_pos] = predict(model, step_data, old_shared[pick], 10, 1)
    ks = {m: ks_2samp(gen[:, j], real[:, j]).statistic for j, m in enumerate(markers)}
    summary = {"edist": energy_distance(gen, real), "mean_ks": float(np.mean(list(ks.values()))),
               "mean_ks_new_only": float(np.mean([ks[m] for m in new_only]))}
    print(json.dumps(summary), {m: round(ks[m], 3) for m in ("CD45RA", "CD34", "CD62L", "CD41", "CD117", "CD14")},
          flush=True)
    src_idx = {m: list(old.var_names).index(m) for m in shared}
    plot(f"new-only flow ({run.name}), sample@10", gen, real, old_full[pick], markers, shared, src_idx,
         out / "markers_newonly_sample_10.png")
    pd.Series(ks, name="ks").to_csv(out / "ks_per_marker.csv")
    pd.Series(summary).to_csv(out / "summary.csv")


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main()
