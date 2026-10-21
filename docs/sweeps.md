# Panel imputation sweep: runbook

Ported from theislab/sckitflow#202. The sweep draws 30 random cells from a 90-cell space (5 input datasets x
`coupling.epsilon` x `coupling.alpha` x `coupling.scale_cost`) with Optuna's `RandomSampler`, and runs each
trial as its own SLURM job through submitit.

| Config | Role |
|---|---|
| `experiment/panel.yaml` | Panel imputation: fused Gromov-Wasserstein, `resnet1d` conditioning, source encoder on |
| `experiment/panel_smoke.yaml` | The same at toy size, `fast_dev_run`, CPU |
| `experiment/panel_sweep.yaml` | The 30-trial Optuna sweep over the panel experiment |
| `experiment/panel_sweep_smoke.yaml` | One trial, to check the Optuna + submitit wiring before a real launch |
| `hydra/launcher/helmholtz.yaml` | The SLURM settings (partition, QoS, GPU, time); swap it for another cluster |

Every trial returns the best value of the metric its checkpoint monitors (`callbacks.model_checkpoint.monitor`,
`test/edist` for the panel data), and Optuna minimises it. A trial that never logs that metric returns `nan` and
counts as failed; the sweep tolerates up to 20 % failed trials (`hydra.sweeper.max_failure_rate`). The sampler is
seeded (`hydra.sweeper.sampler.seed: 0`), so a relaunch draws the same cells.

The panel data splits into train / test only, so, as in #202, the test split also drives model selection (early
stopping, the best checkpoint, and the sweep's choice). Its scores are optimistic: confirm the chosen cell on data
the sweep never saw before reporting it.

## Environment

| Variable | Meaning |
|---|---|
| `FLOWRUN_PANEL_DATA` | Required. The directory holding `adata_concat_seed{42..46}/adata_concat.h5ad` |
| `FLOWRUN_SWEEP_DIR` | Where sweeps go (default `multirun`, relative to the launch directory). Each launch writes a fresh `<FLOWRUN_SWEEP_DIR>/<experiment>/<timestamp>/`, so a relaunch never collides with finished trials |

```bash
export FLOWRUN_PANEL_DATA=/lustre/groups/<group>/<you>/panel_data
export FLOWRUN_SWEEP_DIR=/lustre/groups/<group>/<you>/sweeps
```

## 0. Setup

```bash
uv sync --extra sweep
uv run python -c "import hydra, optuna, hydra_plugins.hydra_optuna_sweeper; print(hydra.__version__, optuna.__version__)"
```

`hydra-core` must be a stable 1.3.x: 1.4 pre-releases break `hydra-optuna-sweeper` at composition
(`Key '_recursive_' not in 'OptunaSweeperConf'`). The lock pins a 1.3 release.

## 1. Prepare the five pre-split inputs

Each input carries `obs["split"]` = `train` / `test` (an 80/20 per-cell split). `fit_cfm` reuses a split column
already in the file and skips its own splitter. The column is required: the panel splitter cannot hold out part of
the single target panel, so without it a run stops with `Nothing was held out` and leaves the file untouched. The
inputs are written by Lorenzo's `prepare_panel_adata.py` (not part of this repository), one per split seed; the
sweep reads them from `$FLOWRUN_PANEL_DATA`.

## 2. Smoke test

**Composition, nothing runs:** `--cfg hydra` prints the resolved Hydra config and exits. Keep stdout and stderr
apart: composition errors go to stderr.

```bash
uv run flowrun-fit-cfm +experiment=panel_sweep_smoke --cfg hydra > /tmp/sweep.out 2> /tmp/sweep.err
echo "exit $?"; cat /tmp/sweep.err
grep -n "_target_" /tmp/sweep.out
```

Expect exit `0` and three targets: `...submitit_launcher.SlurmLauncher`, `...optuna_sweeper.OptunaSweeper` and
`optuna.samplers.RandomSampler`. Leave `-m` out here: Hydra rejects `--cfg` together with `-m`.

**One trial on SLURM (~3 min):**

```bash
uv run flowrun-fit-cfm -m +experiment=panel_sweep_smoke 2>&1 | tee /tmp/cfm_smoke.log
```

The console shows `[HYDRA] #0 : paths.target_adata=...,coupling.epsilon=...`, the cell the sampler drew (seeded,
so the same on every launch). `squeue -u $USER` shows one job. Then check:

```bash
sacct -u $USER --starttime today --format=JobID,State,ExitCode | grep cfm   # COMPLETED 0:0
cat <sweep dir>/trial_0/.hydra/overrides.yaml                                   # the sampled cell
cat <sweep dir>/optimization_results.yaml                                       # best_value: a finite number
```

The same one-trial run, with Hydra's local launcher standing in for SLURM, is `tests/test_panel.py` (marked
`slow`; `pytest -m "not slow"` skips it). With `hydra/launcher=basic` every trial runs in the driver's process, so
they all log to one W&B run; use the SLURM launcher for one run per trial.

## 3. The real sweep

Run the Hydra driver as one small SLURM job, so it survives a disconnect; it submits the 30 trials as GPU jobs
through the `helmholtz` launcher and finishes when they do.

```bash
sbatch --job-name=cfm-sweep --partition=cpu_p --qos=cpu_normal --cpus-per-task=2 --mem=8G --time=48:00:00 \
       --chdir=/path/to/flowrun --output=/tmp/cfm_sweep_%j.log \
       --wrap="uv run flowrun-fit-cfm -m +experiment=panel_sweep"
```

Widen the search without editing the file: `hydra.sweeper.n_trials=90 hydra.sweeper.n_jobs=90`. 90 random draws
from a 90-cell space cover about 63 % of it; for exhaustive coverage use Hydra's basic (grid) sweeper instead.

## 4. Monitor

```bash
squeue -u $USER -o "%.8i %.30j %.8T %.10M %R" | grep cfm
sacct -u $USER --starttime today --format=JobID,State,ExitCode,Elapsed | grep cfm
tail -f <sweep dir>/<trial>/logs/csv/*/*/metrics.csv   # <name>/version_<n>/metrics.csv
```

If trials run one at a time, SLURM is throttling them, not Hydra: `1 RUNNING, N PENDING` is usually the QoS
`MaxJobsPU` cap (`sacctmgr show qos <qos> format=Name,MaxJobsPU`); switch the QoS in
`hydra/launcher/helmholtz.yaml` or ask for a higher cap. All `PENDING` with `ReqNodeNotAvail`: the `constraint`
there cannot be met; remove it.

## 5. Collect results

Optuna writes the best cell to `<sweep dir>/optimization_results.yaml`. Each trial's `save/` is a full model
(`sckitflow.Model.from_pretrained`), and `.hydra/overrides.yaml` holds its sampled cell. Sync the offline W&B runs
with `wandb sync --sync-all <sweep dir>`; they are grouped as `panel-adata-coupling-sweep`.

## 6. Extending the search space

Add axes to `hydra.sweeper.params` in `panel_sweep.yaml`, one at a time, and raise `n_trials` with the space:

```yaml
module.state_encoder_output_dim: choice(16, 32)
module.condition_encoder_output_dim: choice(32, 64)
```

For wide spaces drop the `override /hydra/sweeper/sampler: random` line to use Optuna's TPE sampler, which
concentrates trials around good regions.

## 7. Troubleshooting

| Symptom | Fix |
|---|---|
| `Key '_recursive_' not in 'OptunaSweeperConf'` at composition | `hydra-core` is a 1.4 pre-release; use the locked 1.3 (`uv sync --extra sweep`) |
| `Key 'infer_relative_search_space' not in 'SamplerConfig'` | The sampler reached Optuna uninstantiated: select it with `override /hydra/sweeper/sampler: random` only, never with a manual `sampler:` block |
| `Could not find 'hydra/sweeper/optuna'` | The `sweep` extra is missing: `uv sync --extra sweep` |
| `Key 'partition' not in 'BasicLauncherConf'` | The SLURM settings belong to the `helmholtz` launcher; another launcher needs its own config |
| `Hydra15MigrationWarning` about `version_base` | Harmless on hydra-core 1.3 |
| `File exists: .../save` on a retried trial | A preempted trial left a partial directory: remove it, or set `run.allow_overwrite=true` |
| `Nothing was held out` | The input has no prepared `obs["split"]`; write the 80/20 split when preparing the data |
| `oc.env` error naming `FLOWRUN_PANEL_DATA` | Export it (see Environment) |
| `Sinkhorn did not converge` | Harmless unless the metric plateaus; try a larger `coupling.epsilon` |
