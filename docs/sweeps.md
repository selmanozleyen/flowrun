# Flow cytometry panel alignment sweeps: runbook

Ported from theislab/sckitflow#202. Two sweeps, each 30 random draws with Optuna's `RandomSampler`, every trial its
own SLURM job through submitit:

- **fused**: fused Gromov-Wasserstein, over 5 split seeds x `coupling.epsilon` x `coupling.alpha` x
  `coupling.scale_cost` (90 cells);
- **linear**: entropic linear OT (Sinkhorn) on the shared markers only, over 5 split seeds x
  `coupling.reg` x `coupling.scale_cost` (30 cells). The baseline for the fused one.

| Config | Role |
|---|---|
| `experiment/flow_cyto_panel_alignment.yaml` | Panel alignment: fused Gromov-Wasserstein, ResNet conditioning, source encoder on |
| `experiment/flow_cyto_panel_alignment_smoke.yaml` | The same at toy size, `fast_dev_run`, CPU |
| `experiment/flow_cyto_panel_alignment_sweep.yaml` | What both sweeps share: the data axis, launcher, sampler, output dirs |
| `experiment/flow_cyto_panel_alignment_sweep_fused.yaml` | The fused GW sweep |
| `experiment/flow_cyto_panel_alignment_sweep_linear.yaml` | The linear OT sweep |
| `experiment/flow_cyto_panel_alignment_sweep_smoke.yaml` | One trial of the fused sweep, to check the Optuna + submitit wiring before a real launch |
| a project's `hydra/launcher/<site>.yaml` | The SLURM settings (partition, QoS, GPU, time), e.g. `helmholtz` in theislab/flow-cytometry-imputation |

Every trial returns the best value of the metric its checkpoint monitors (`callbacks.model_checkpoint.monitor`,
`val/edist` for the panel data), and Optuna minimises it. A trial that never logs that metric raises and
counts as failed; the sweep tolerates up to 20 % failed trials (`hydra.sweeper.max_failure_rate`). The sampler is
seeded (`hydra.sweeper.sampler.seed: 0`), so a relaunch draws the same cells.

Each run splits the new panel's cells at random into train / val / test (70 / 10 / 20, `splitter.type:
random_cells`); the old panel is every split's source. `val/edist` drives model selection (early stopping, the best
checkpoint, and the sweep's choice), so `test` stays unseen until the final evaluation. A seed's split is written to
`obs["split_<seed>"]` of the inputs the first time and reused after, so every model of a seed shares it.

## Environment

| Variable | Meaning |
|---|---|
| `FLOWRUN_SWEEP_DIR` | Where sweeps go (default `multirun`, relative to the launch directory). Each launch writes a fresh `<FLOWRUN_SWEEP_DIR>/<experiment>/<timestamp>/`, so a relaunch never collides with finished trials |

```bash
export FLOWRUN_SWEEP_DIR=/lustre/groups/<group>/<you>/sweeps
```

## 0. Setup

```bash
uv sync --extra cu126   # V100s in gpu_p; cu130 for newer GPUs
uv run python -c "import hydra, optuna, hydra_plugins.hydra_optuna_sweeper; print(hydra.__version__, optuna.__version__)"
```

`hydra-core` must be a stable 1.3.x: 1.4 pre-releases break `hydra-optuna-sweeper` at composition
(`Key '_recursive_' not in 'OptunaSweeperConf'`). The lock pins a 1.3 release.

## 1. Prepare the panels

Once per dataset, a project renames synonyms, labels each file as a panel (`obs` and its one-hot `uns` table) and
writes one zarr per panel, then sets `paths.target_adata` to them; for the CITE-seq panels that is
`python -m flow_cytometry_imputation.prepare` and the `citeseq*` experiments in theislab/flow-cytometry-imputation.
`fit_cfm` takes the shared and own markers by name from the state (`lin_from_state` / `quad_from_state`).

## 2. Smoke test

**Composition, nothing runs:** `--cfg hydra` prints the resolved Hydra config and exits. Keep stdout and stderr
apart: composition errors go to stderr.

```bash
uv run flowrun-fit-cfm +experiment=flow_cyto_panel_alignment_sweep_smoke --cfg hydra > /tmp/sweep.out 2> /tmp/sweep.err
echo "exit $?"; cat /tmp/sweep.err
grep -n "_target_" /tmp/sweep.out
```

Expect exit `0` and three targets: `...submitit_launcher.SlurmLauncher`, `...optuna_sweeper.OptunaSweeper` and
`optuna.samplers.RandomSampler`. Leave `-m` out here: Hydra rejects `--cfg` together with `-m`.

**One trial on SLURM (~3 min):**

```bash
uv run flowrun-fit-cfm -m +experiment=flow_cyto_panel_alignment_sweep_smoke 2>&1 | tee /tmp/cfm_smoke.log
```

The console shows `[HYDRA] #0 : paths.target_adata=...,coupling.epsilon=...`, the cell the sampler drew (seeded,
so the same on every launch). `squeue -u $USER` shows one job. Then check:

```bash
sacct -u $USER --starttime today --format=JobID,State,ExitCode | grep cfm   # COMPLETED 0:0
cat <sweep dir>/trial_0/.hydra/overrides.yaml                                   # the sampled cell
cat <sweep dir>/optimization_results.yaml                                       # best_value: a finite number
```

The same one-trial run, with Hydra's local launcher standing in for SLURM, is `tests/test_flow_cyto_panel_alignment.py` (marked
`slow`; `pytest -m "not slow"` skips it). With `hydra/launcher=basic` every trial runs in the driver's process, so
each trial still gets its own W&B run (`fit_cfm` finishes its run).

## 3. The real sweep

Run the Hydra driver as one small SLURM job, so it survives a disconnect; it submits the 30 trials as GPU jobs
through the project's SLURM launcher and finishes when they do.

```bash
sbatch --job-name=cfm-sweep --partition=cpu_p --qos=cpu_normal --cpus-per-task=2 --mem=8G --time=48:00:00 \
       --chdir=/path/to/flowrun --output=/tmp/cfm_sweep_%j.log \
       --wrap="uv run flowrun-fit-cfm -m +experiment=flow_cyto_panel_alignment_sweep_fused"
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
the project's launcher config or ask for a higher cap. All `PENDING` with `ReqNodeNotAvail`: the `constraint`
there cannot be met; remove it.

## 5. Collect results

Optuna writes the best cell to `<sweep dir>/optimization_results.yaml`. Each trial's `save/` is a full model
(`sckitflow.Model.from_pretrained`), and `.hydra/overrides.yaml` holds its sampled cell. Sync the offline W&B runs
with `wandb sync --sync-all <sweep dir>`; each sweep is its own W&B group, named after it.

## 6. Extending the search space

Add axes to `hydra.sweeper.params` in `flow_cyto_panel_alignment_sweep_fused.yaml` (or `_linear`), one at a time, and raise `n_trials` with the space:

```yaml
module.state_encoder.output_dim: choice(16, 32)
module.time_encoder.output_dim: choice(16, 32)
```

For wide spaces drop the `override /hydra/sweeper/sampler: random` line to use Optuna's TPE sampler, which
concentrates trials around good regions.

## 7. Troubleshooting

| Symptom | Fix |
|---|---|
| `Key '_recursive_' not in 'OptunaSweeperConf'` at composition | `hydra-core` is a 1.4 pre-release; use the locked 1.3 (`uv sync --locked`) |
| `Key 'infer_relative_search_space' not in 'SamplerConfig'` | The sampler reached Optuna uninstantiated: select it with `override /hydra/sweeper/sampler: random` only, never with a manual `sampler:` block |
| `Could not find 'hydra/sweeper/optuna'` | Not flowrun's environment: `uv sync`, then `uv run` |
| `Key 'partition' not in 'BasicLauncherConf'` | SLURM settings need a SLURM launcher: select the project's (`hydra/launcher=<site>`) |
| `Hydra15MigrationWarning` about `version_base` | Harmless on hydra-core 1.3 |
| `File exists: .../save` on a retried trial | A preempted trial left a partial directory: remove it, or set `run.allow_overwrite=true` |
| `Sinkhorn did not converge` | Harmless unless the metric plateaus; try a larger `coupling.epsilon` |
