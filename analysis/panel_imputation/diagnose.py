"""Why are bimodal markers smeared? Mean over samples vs one sample, and ODE steps, for one saved panel model.

    python diagnose.py <run dir (has .hydra/ and save/)> <out dir> [n_cells]

Loads the data and model as flowrun does, maps old-panel cells to the new panel under several inference settings,
and compares each per marker against the new panel's held-out test cells (KS, energy distance, per-marker plots).
"""

import json
import sys
import time
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf
from scipy.stats import ks_2samp, spearmanr

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from flowrun.fit_cfm import as_list, enable_zarrs, read_zarr_input  # noqa: E402
from flowrun.panels import _state_values, from_state  # noqa: E402
from sckitflow import Model  # noqa: E402
from sckitflow.core.methods import ODEInference  # noqa: E402
from sckitflow.data import FlowDataModule  # noqa: E402

# name -> (n_steps, n_samples): n_samples > 1 gives X as the mean over samples, 1 gives one sample per cell
SETTINGS = {"mean100@10": (10, 100), "sample@10": (10, 1), "sample@50": (50, 1), "sample@200": (200, 1)}
FOCUS = ["CD45RA", "CD34", "CD62L", "CD14", "CD41", "HLA_DR"]


def energy_distance(a: np.ndarray, b: np.ndarray, n: int = 5000, seed: int = 0) -> float:
    rng = np.random.default_rng(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    a = torch.as_tensor(a[rng.choice(len(a), min(n, len(a)), replace=False)], dtype=torch.float32, device=device)
    b = torch.as_tensor(b[rng.choice(len(b), min(n, len(b)), replace=False)], dtype=torch.float32, device=device)
    return (2 * torch.cdist(a, b).mean() - torch.cdist(a, a).mean() - torch.cdist(b, b).mean()).item()


def predict(model, step_data, source_rows: np.ndarray, n_steps: int, n_samples: int, seed: int = 0) -> np.ndarray:
    config = model.inference_method.config.model_copy(update={"n_steps": n_steps, "n_samples": n_samples})
    method = ODEInference(model.module, config, dims=model.dims)
    out = []
    for start in range(0, len(source_rows), 2048):
        rows = torch.as_tensor(source_rows[start : start + 2048], dtype=torch.float32, device="cuda")
        data = {**step_data, "source_state": rows, "n_obs": len(rows), "target_state": None}
        generator = torch.Generator(device="cuda").manual_seed(seed + start)
        with torch.no_grad():
            out.append(method.predict(data, generator=generator).X.detach().cpu().numpy())
    return np.concatenate(out)


def shared_correlation(gen: np.ndarray, source: np.ndarray, markers, shared, src_idx) -> pd.Series:
    """Per shared marker, Spearman correlation of each cell's generated value with its own measured source value.

    1 means the cell's own value carries through (up to a monotone recalibration); 0 means the source is ignored.
    """
    return pd.Series({m: spearmanr(gen[:, markers.index(m)], source[:, src_idx[m]]).statistic for m in shared},
                     name="spearman(gen, own source)")


def plot(name: str, gen: np.ndarray, real: np.ndarray, old: np.ndarray, markers, shared, src_idx, path: Path) -> None:
    ncol = 4
    fig, axes = plt.subplots(int(np.ceil(len(markers) / ncol)), ncol, figsize=(16, 3.2 * np.ceil(len(markers) / ncol)))
    for ax, (j, m) in zip(axes.flat, enumerate(markers), strict=False):
        lo, hi = np.percentile(np.concatenate([real[:, j], gen[:, j]]), [0.1, 99.9])
        bins = np.linspace(lo - 0.5, hi + 0.5, 80)
        ax.hist(real[:, j], bins=bins, density=True, histtype="step", color="tab:blue", label="real (new, test)")
        ax.hist(gen[:, j], bins=bins, density=True, histtype="step", color="tab:red", label="generated")
        title = f"{m} ({'shared' if m in shared else 'own'})\nKS(gen,real)={ks_2samp(gen[:, j], real[:, j]).statistic:.3f}"
        if m in shared:
            o = old[:, src_idx[m]]
            ax.hist(o, bins=bins, density=True, histtype="step", color="tab:green", label="old (source)")
            title += f"\nKS(old,real)={ks_2samp(o, real[:, j]).statistic:.3f}"
        ax.set_title(title, fontsize=8)
    for ax in list(axes.flat)[len(markers) :]:
        ax.axis("off")
    axes.flat[0].legend(fontsize=7)
    fig.suptitle(f"Per-marker distributions, {name}", y=1.0)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main() -> None:
    run, out = Path(sys.argv[1]), Path(sys.argv[2])
    n_cells = int(sys.argv[3]) if len(sys.argv) > 3 else 20000
    out.mkdir(parents=True, exist_ok=True)
    enable_zarrs()
    cfg = OmegaConf.load(run / ".hydra" / "config.yaml")
    adatas = [read_zarr_input(Path(p), cfg) for p in as_list(cfg.paths.target_adata)]
    from_state(adatas, OmegaConf.to_container(cfg.datamanager, resolve=True))
    model = Model.from_pretrained(run / "save", adatas)
    model.module.to("cuda").eval()
    datamodule = FlowDataModule.from_adata(adatas, model.manager, n_train_steps=1, batch_size=256)
    step_data, _ = next(iter(datamodule.dm.get_eval_loader(adatas, max_per_group=1)))
    print("trained inference config:", model.inference_method.config, flush=True)

    control = str(cfg.datamanager.control_values_dict.panel)
    is_source = [str(a.obs["dataset"].iloc[0]) == control for a in adatas]
    source, target = adatas[is_source.index(True)], adatas[is_source.index(False)]
    state = cfg.datamanager.target_sample_rep
    split = cfg.splitter.split_key if "$" not in str(cfg.splitter.split_key) else f"split_{cfg.run.splitter_seed}"
    source_all = _state_values(source, state)
    real = _state_values(target, state)[(target.obs[split] == "test").to_numpy()]
    markers = list(target.var_names)
    shared = sorted(set(source.var_names) & set(markers))
    src_idx = {m: list(source.var_names).index(m) for m in shared}
    rows = np.sort(np.random.default_rng(0).choice(source.n_obs, size=min(n_cells, source.n_obs), replace=False))
    print(f"source {source.n_obs} cells, target test {len(real)} cells, {len(markers)} markers, {len(shared)} shared",
          flush=True)

    ks_rows, summary = [], []
    names = sys.argv[4].split(",") if len(sys.argv) > 4 else list(SETTINGS)  # a subset of SETTINGS
    for name in names:
        n_steps, n_samples = SETTINGS[name]
        t0 = time.perf_counter()
        gen = predict(model, step_data, source_all[rows], n_steps, n_samples)
        secs = time.perf_counter() - t0
        for j, m in enumerate(markers):
            ks_rows.append({"setting": name, "marker": m, "ks": ks_2samp(gen[:, j], real[:, j]).statistic,
                            "shared": m in shared})
        ed = energy_distance(gen, real)
        summary.append({"setting": name, "n_steps": n_steps, "n_samples": n_samples, "edist": ed,
                        "mean_ks": float(np.mean([r["ks"] for r in ks_rows if r["setting"] == name])), "secs": secs})
        print(json.dumps(summary[-1]), flush=True)
        plot(name, gen, real, source_all[rows], markers, shared, src_idx, out / f"markers_{name.replace('@', '_')}.png")
        if n_samples == 1:
            corr = shared_correlation(gen, source_all[rows], markers, shared, src_idx)
            corr.to_csv(out / f"shared_corr_{name.replace('@', '_')}.csv")
            print(f"{name} spearman(gen, own source): mean {corr.mean():.3f}", corr.round(3).to_dict(), flush=True)
    for m in shared:
        ks_rows.append({"setting": "old (source)", "marker": m, "shared": True,
                        "ks": ks_2samp(source_all[rows][:, src_idx[m]], real[:, markers.index(m)]).statistic})

    ks = pd.DataFrame(ks_rows)
    ks.to_csv(out / "ks_per_marker.csv", index=False)
    pd.DataFrame(summary).to_csv(out / "summary.csv", index=False)
    table = ks.pivot(index="marker", columns="setting", values="ks")
    print(table.loc[[m for m in FOCUS if m in table.index]].round(3).to_string(), flush=True)
    print(table.mean().round(4).to_string(), flush=True)


if __name__ == "__main__":
    main()
