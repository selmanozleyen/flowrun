# Panel imputation analysis: running manual

How to reproduce the numbers in [`docs/panel_imputation.md`](../../docs/panel_imputation.md) on the HMGU cluster.

## Setup

The jobs run against the theislab/flow-cytometry-imputation (`fci`) lock, which pins flowrun and sckitflow. Clone it
once into a workspace dir, and put this folder's scripts next to it:

```bash
WS=/ictstr01/groups/ml01/workspace/$USER/fci-diagnose
git clone git@github.com:theislab/flow-cytometry-imputation.git $WS/fci
cp analysis/panel_imputation/*.{py,sh} $WS/
```

Every job copies `$WS/fci` to `/localscratch`, builds the env there with `uv sync --frozen` (uv cache and `TMPDIR`
on local scratch too, never Lustre), and writes results back to `$WS`.

Inputs:

| path | what |
|---|---|
| `/lustre/groups/ml01/datasets/$USER/citeseq_panel/prepared` | the two prepared panels (`OH_OH0013_OLD_UMAP_annotated_red.zarr`, `260331_OH013_New_Panel_annotated_red.zarr`) with `split_42` in obs |
| a finished `fci-fit` run dir (`.hydra/`, `save/`) | `diagnose.py` loads its model; the GMM scripts only read its data config |

The run dirs in the job scripts are the ones used for the doc. Edit them to point at yours.

## Jobs

Submit from a login node with `W=$WS`. CPU jobs use the interactive partition (8 CPUs, 30G, the QoS cap); the
GPU ones are V100s, so the lock's torch is cu126.

| step | script | submit |
|---|---|---|
| diagnose a flow run (sampling, steps, source correlation) | `run_diag.sh` → `diagnose.py` | `sbatch -p interactive_gpu_p -q interactive_gpu_short --gres=gpu:1 -c 6 --mem=16G -t 1:00:00 -o $W/diag.log $W/run_diag.sh` |
| capacity grid (base, wide, src64, wide + src64; eps 1), 4 runs on one GPU, then `diagnose.py` | `run_capacity.sh` | `sbatch -p gpu_p -q gpu_normal --gres=gpu:1 -c 12 --mem=64G -t 20:00:00 -o $W/capacity.log $W/run_capacity.sh` |
| coupling sharpness (eps 1, 0.1, 0.05), wide + src64, then fallback count and `diagnose.py` | `run_sharp.sh` | `sbatch -p interactive_gpu_p -q interactive_gpu_short --gres=gpu:1 -c 8 --mem=20G -t 4:00:00 -o $W/sharp.log $W/run_sharp.sh` |
| GMM baseline, old cells imputed | `run_gmm.sh` → `gmm_baseline.py` | `sbatch -p interactive_cpu_p -q interactive_cpu_short -c 8 --mem=30G -t 3:00:00 -o $W/gmm.log $W/run_gmm.sh` |
| GMM per cell vs kNN and linear | `run_gmm_percell.sh` → `gmm_percell.py` | `sbatch -p interactive_cpu_p -q interactive_cpu_short -c 8 --mem=30G -t 2:00:00 -o $W/gmm_percell.log $W/run_gmm_percell.sh` |
| obs annotations and K sweep | `run_obs_k.sh` → `obs_and_k.py` | `sbatch -p interactive_cpu_p -q interactive_cpu_short -c 8 --mem=30G -t 2:00:00 -o $W/obs_k.log $W/run_obs_k.sh` |
| new-only panel copies + smoke run | `run_newonly_prep.sh` → `prep_newonly.py` | `sbatch -p interactive_cpu_p -q interactive_cpu_short -c 8 --mem=30G -t 1:00:00 -o $W/newonly_prep.log $W/run_newonly_prep.sh` |
| train the new-only flow | `run_newonly_train.sh` (overrides in `newonly_args.sh`) | `sbatch -p interactive_gpu_p -q interactive_gpu_short --gres=gpu:1 -c 6 --mem=16G -t 5:00:00 -o $W/newonly_train.log $W/run_newonly_train.sh` |
| evaluate it per cell and on old cells | `run_newonly_eval.sh` → `newonly_eval.py` | `sbatch -p interactive_gpu_p -q interactive_gpu_short --gres=gpu:1 -c 6 --mem=16G -t 0:30:00 -o $W/newonly_eval.log $W/run_newonly_eval.sh` |

Order: `gmm_percell` before `newonly_eval` (the eval joins its CSV), and `newonly_prep` before `newonly_train`.
The train script prints `TRAIN_DONE <run dir>`; put that dir into `run_newonly_eval.sh`.

Outputs, each in its own `$WS/out_*` dir: `summary.csv`, `ks_per_marker.csv` or `percell_new_only.csv`, and the
per-marker density plots (`markers_*.png`).

## Gotchas

- Interactive CPU jobs: memory above 8 × 2.5G raises the CPU count past the QoS cap (`QOSMaxCpuPerJobLimit`).
- Compute nodes have no `rsync`; the scripts use `cp -a`.
- `prep_newonly.py` writes with `annbatch.io.write_sharded`. anndata's own zarr writers give a chunk layout the
  annbatch loader cannot read.

## Results

The CSVs behind the doc's tables are in [`results/`](results). The full outputs (plots, logs, checkpoints) stay on
the cluster, readable by the ml01 group:

- analysis outputs, plots and logs: `/ictstr01/groups/ml01/workspace/selman.ozleyen/fci-diagnose/out_*`, `*.log`
- runs (config, checkpoints, saved models): `/lustre/groups/ml01/workspace/selman.ozleyen/fci_runs/`
- prepared panels: `/lustre/groups/ml01/datasets/selman.ozleyen/citeseq_panel/` (`prepared`, `prepared_newonly`)

| dir | from | run |
|---|---|---|
| `flow_small_grid` | `run_diag.sh` | `fci_runs/grid_2026-10-22_07-43-14/3`, the 4 sampling settings |
| `flow_wide_src64` | `diagnose.py` on a mid-training checkpoint | capacity grid, wide + src64 |
| `capacity_*` | `run_capacity.sh` | `fci_runs/capacity_2026-10-26_19-35/<variant>` |
| `sharp_eps*` | `run_sharp.sh` | `fci_runs/sharp_2026-10-26_19-59/eps<eps>` (eps 0.05 killed: always fell back) |
| `gmm_baseline` | `run_gmm.sh` | GMM K = 20, conditional and unconditional, old cells |
| `gmm_percell` | `run_gmm_percell.sh` | GMM, kNN, linear per cell in the new panel |
| `k_sweep` | `run_obs_k.sh` | K sweep, mean over 2 seeds |
| `flow_newonly` | `run_newonly_eval.sh` | `fci_runs/newonly_2026-10-26_21-03` |
