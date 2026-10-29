"""Per-cell check of imputing the new-only markers from the shared ones, inside the new panel (ground truth known).

    python gmm_percell.py <run dir (for the data config)> <out dir>

New-panel train cells fit the models; on the new-panel test cells the 12 new-only markers are hidden and predicted
from the 16 shared ones. Per marker: R^2 and Pearson r of the GMM's conditional mean E[n | s], Pearson r of one
conditional sample, and KS of the sample against the truth; linear regression and kNN on the shared markers as
baselines.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from omegaconf import OmegaConf
from scipy.stats import ks_2samp, multivariate_normal, pearsonr
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import KNeighborsRegressor

from flowrun.fit_cfm import as_list, enable_zarrs, read_zarr_input
from flowrun.panels import _state_values, from_state


def gmm_conditional(gmm: GaussianMixture, s: np.ndarray, s_idx, n_idx, rng: np.random.Generator):
    """E[n | s] and one draw from p(n | s), per row; and the mean top responsibility."""
    K = gmm.n_components
    log_r = np.empty((len(s), K))
    means = np.empty((K, len(s), len(n_idx)))
    chols = []
    for k in range(K):
        mu, S = gmm.means_[k], gmm.covariances_[k]
        S_ss, S_ns, S_nn = S[np.ix_(s_idx, s_idx)], S[np.ix_(n_idx, s_idx)], S[np.ix_(n_idx, n_idx)]
        gain = S_ns @ np.linalg.inv(S_ss)
        means[k] = mu[n_idx] + (s - mu[s_idx]) @ gain.T
        chols.append(np.linalg.cholesky(S_nn - gain @ S_ns.T + 1e-9 * np.eye(len(n_idx))))
        log_r[:, k] = np.log(gmm.weights_[k]) + multivariate_normal(mu[s_idx], S_ss, allow_singular=True).logpdf(s)
    r = np.exp(log_r - log_r.max(1, keepdims=True))
    r /= r.sum(1, keepdims=True)
    mean = np.einsum("ik,kij->ij", r, means)
    z = (r.cumsum(1) > rng.random((len(s), 1))).argmax(1)
    sample = means[z, np.arange(len(s))]
    noise = rng.standard_normal((len(s), len(n_idx)))
    for k in range(K):
        rows = z == k
        sample[rows] += noise[rows] @ chols[k].T
    return mean, sample, float(r.max(1).mean())


def main() -> None:
    run, out = Path(sys.argv[1]), Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    enable_zarrs()
    cfg = OmegaConf.load(run / ".hydra" / "config.yaml")
    adatas = [read_zarr_input(Path(p), cfg) for p in as_list(cfg.paths.target_adata)]
    from_state(adatas, OmegaConf.to_container(cfg.datamanager, resolve=True))
    is_source = [str(a.obs["dataset"].iloc[0]) == str(cfg.datamanager.control_values_dict.panel) for a in adatas]
    source, target = adatas[is_source.index(True)], adatas[is_source.index(False)]
    x = _state_values(target, cfg.datamanager.target_sample_rep).astype(np.float64)
    labels = target.obs[f"split_{cfg.run.splitter_seed}"].astype(str).to_numpy()
    train, test = x[labels == "train"], x[labels == "test"]
    markers = list(target.var_names)
    shared = set(source.var_names)
    s_idx = [j for j, m in enumerate(markers) if m in shared]
    n_idx = [j for j, m in enumerate(markers) if m not in shared]
    print(f"train {len(train)}, test {len(test)}; predict {len(n_idx)} new-only from {len(s_idx)} shared", flush=True)

    rng = np.random.default_rng(0)
    fit_rows = rng.choice(len(train), size=min(60000, len(train)), replace=False)
    gmm = GaussianMixture(20, covariance_type="full", reg_covar=1e-4, random_state=0, max_iter=300).fit(train[fit_rows])
    s_test, n_test = test[:, s_idx], test[:, n_idx]
    gmm_mean, gmm_sample, resp = gmm_conditional(gmm, s_test, s_idx, n_idx, rng)
    print(f"GMM K=20, mean top responsibility on test cells {resp:.3f}", flush=True)
    lin = LinearRegression().fit(train[fit_rows][:, s_idx], train[fit_rows][:, n_idx]).predict(s_test)
    knn = KNeighborsRegressor(n_neighbors=30, n_jobs=-1).fit(train[fit_rows][:, s_idx], train[fit_rows][:, n_idx])
    knn = knn.predict(s_test)

    rows = []
    for i, j in enumerate(n_idx):
        t = n_test[:, i]
        rows.append({
            "marker": markers[j],
            "R2 gmm mean": r2_score(t, gmm_mean[:, i]),
            "R2 knn": r2_score(t, knn[:, i]),
            "R2 linear": r2_score(t, lin[:, i]),
            "r gmm mean": pearsonr(t, gmm_mean[:, i]).statistic,
            "r gmm sample": pearsonr(t, gmm_sample[:, i]).statistic,
            "KS gmm sample": ks_2samp(t, gmm_sample[:, i]).statistic,
        })
    table = pd.DataFrame(rows).set_index("marker").sort_values("R2 gmm mean")
    table.loc["mean"] = table.mean()
    table.to_csv(out / "percell_new_only.csv")
    print(table.round(3).to_string(), flush=True)


if __name__ == "__main__":
    main()
