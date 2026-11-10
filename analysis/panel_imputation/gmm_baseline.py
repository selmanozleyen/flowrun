"""Conditional Gaussian mixture baseline for panel imputation.

    python gmm_baseline.py <run dir (for the data config)> <out dir>

A GMM over all new-panel markers is fitted on the new panel's train cells. An old-panel cell's shared markers are
quantile-matched to new-panel units; its new-only markers are drawn from the GMM conditioned on them:
p(n | s) = sum_k p(k | s) N(n; mu_n + S_ns S_ss^-1 (s - mu_s), S_nn - S_ns S_ss^-1 S_sn). Scored like diagnose.py on
the same 20k old-panel cells against the new panel's test cells.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from omegaconf import OmegaConf
from scipy.stats import ks_2samp, multivariate_normal
from sklearn.mixture import GaussianMixture

sys.path.insert(0, str(Path(__file__).parent))
from diagnose import FOCUS, energy_distance, plot  # noqa: E402
from flowrun.fit_cfm import as_list, enable_zarrs, read_zarr_input  # noqa: E402
from flowrun.panels import _state_values, from_state  # noqa: E402


def quantile_match(x: np.ndarray, reference: np.ndarray, n_q: int = 1001) -> np.ndarray:
    """Map ``x`` monotonically so its distribution matches ``reference`` (per column)."""
    q = np.linspace(0, 1, n_q)
    out = np.empty_like(x)
    for j in range(x.shape[1]):
        out[:, j] = np.interp(x[:, j], np.quantile(x[:, j], q), np.quantile(reference[:, j], q))
    return out


def conditional_sample(gmm: GaussianMixture, s: np.ndarray, s_idx, n_idx, rng: np.random.Generator) -> np.ndarray:
    """One draw of the new-only coordinates per row, from the GMM conditioned on the shared coordinates ``s``."""
    K = gmm.n_components
    log_r = np.empty((len(s), K))
    cond = []
    for k in range(K):
        mu, S = gmm.means_[k], gmm.covariances_[k]
        S_ss, S_ns, S_nn = S[np.ix_(s_idx, s_idx)], S[np.ix_(n_idx, s_idx)], S[np.ix_(n_idx, n_idx)]
        gain = S_ns @ np.linalg.inv(S_ss)
        cond.append((mu[n_idx], mu[s_idx], gain, S_nn - gain @ S_ns.T))
        log_r[:, k] = np.log(gmm.weights_[k]) + multivariate_normal(mu[s_idx], S_ss, allow_singular=True).logpdf(s)
    r = np.exp(log_r - log_r.max(1, keepdims=True))
    r /= r.sum(1, keepdims=True)
    z = (r.cumsum(1) > rng.random((len(s), 1))).argmax(1)  # a component per row, by its responsibility
    out = np.empty((len(s), len(n_idx)))
    for k in range(K):
        rows = z == k
        if rows.any():
            mu_n, mu_s, gain, S_c = cond[k]
            mean = mu_n + (s[rows] - mu_s) @ gain.T
            out[rows] = mean + rng.multivariate_normal(np.zeros(len(n_idx)), S_c, size=rows.sum())
    print(f"  responsibilities: mean max {r.max(1).mean():.3f} (1 = the shared markers fix the component)", flush=True)
    return out


def main() -> None:
    run, out = Path(sys.argv[1]), Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    enable_zarrs()
    cfg = OmegaConf.load(run / ".hydra" / "config.yaml")
    adatas = [read_zarr_input(Path(p), cfg) for p in as_list(cfg.paths.target_adata)]
    from_state(adatas, OmegaConf.to_container(cfg.datamanager, resolve=True))
    is_source = [str(a.obs["dataset"].iloc[0]) == str(cfg.datamanager.control_values_dict.panel) for a in adatas]
    source, target = adatas[is_source.index(True)], adatas[is_source.index(False)]
    rep, split = cfg.datamanager.target_sample_rep, f"split_{cfg.run.splitter_seed}"
    source_all, target_all = _state_values(source, rep), _state_values(target, rep)
    labels = target.obs[split].astype(str).to_numpy()
    train, real = target_all[labels == "train"], target_all[labels == "test"]
    markers = list(target.var_names)
    shared = sorted(set(source.var_names) & set(markers))
    s_idx = [markers.index(m) for m in shared]
    n_idx = [j for j, m in enumerate(markers) if m not in shared]
    src_idx = {m: list(source.var_names).index(m) for m in shared}
    rows = np.sort(np.random.default_rng(0).choice(source.n_obs, size=min(20000, source.n_obs), replace=False))
    print(f"train {len(train)}, test {len(real)}, {len(shared)} shared, {len(n_idx)} new-only", flush=True)

    rng = np.random.default_rng(0)
    fit_rows = rng.choice(len(train), size=min(60000, len(train)), replace=False)
    best = None
    for K in (10, 20, 30, 40):
        t0 = time.perf_counter()
        gmm = GaussianMixture(K, covariance_type="full", reg_covar=1e-4, random_state=0, max_iter=300)
        gmm.fit(train[fit_rows])
        bic = gmm.bic(train[fit_rows])
        print(f"K={K}: BIC {bic:.0f} ({time.perf_counter() - t0:.0f}s, converged {gmm.converged_})", flush=True)
        if best is None or bic < best[0]:
            best = (bic, K, gmm)
    _, K, gmm = best
    print(f"chosen K={K}", flush=True)

    # shared markers: the old panel's values in new-panel units, by quantile matching on all cells of each panel
    s_new = quantile_match(source_all[:, [src_idx[m] for m in shared]], train[:, s_idx])[rows]
    results, ks_rows = [], []
    for name in ("gmm_conditional", "gmm_unconditional"):
        gen = np.empty((len(rows), len(markers)))
        if name == "gmm_conditional":
            gen[:, s_idx] = s_new
            gen[:, n_idx] = conditional_sample(gmm, s_new, s_idx, n_idx, rng)
        else:
            gen = gmm.sample(len(rows))[0][rng.permutation(len(rows))]
        ks = {m: ks_2samp(gen[:, j], real[:, j]).statistic for j, m in enumerate(markers)}
        ks_rows += [{"setting": name, "marker": m, "ks": v, "shared": m in shared} for m, v in ks.items()]
        results.append({"setting": name, "K": K, "edist": energy_distance(gen, real),
                        "mean_ks": float(np.mean(list(ks.values()))),
                        "mean_ks_new_only": float(np.mean([ks[markers[j]] for j in n_idx]))})
        print(json.dumps(results[-1]), flush=True)
        plot(f"{name} (K={K})", gen, real, source_all[rows], markers, shared, src_idx, out / f"markers_{name}.png")
    table = pd.DataFrame(ks_rows).pivot(index="marker", columns="setting", values="ks")
    table.to_csv(out / "ks_per_marker.csv")
    pd.DataFrame(results).to_csv(out / "summary.csv", index=False)
    print(table.loc[[m for m in FOCUS if m in table.index]].round(3).to_string(), flush=True)


if __name__ == "__main__":
    main()
