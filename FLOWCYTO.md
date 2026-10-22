# Project guidelines.

---

## 0. Preparation.

In the preparation, we set all the environmental variables and create the necessary directories.

```bash
# The root folder for the project
export FLOWCYTO_ROOT="/lustre/groups/ml01/projects/FlowCytometry";
mkdir -p "${FLOWCYTO_ROOT}";

# The folder where the raw data will be downloaded.
mkdir -p "${FLOWCYTO_ROOT}/raw_data";

# The base folder for all kind of outputs from the pipeline.
mkdir -p "${FLOWCYTO_ROOT}/out";

# Where we will dump the prepared data.
mkdir -p "${FLOWCYTO_ROOT}/out/data";


# All the models and runs will be saved here
mkdir -p "${FLOWCYTO_ROOT}/out/dumps";
mkdir -p "${FLOWCYTO_ROOT}/out/dumps/logs";
mkdir -p "${FLOWCYTO_ROOT}/out/dumps/runs";
mkdir -p "${FLOWCYTO_ROOT}/out/dumps/runs_smoke";
mkdir -p "${FLOWCYTO_ROOT}/out/dumps/sweeps";
mkdir -p "${FLOWCYTO_ROOT}/out/dumps/sweeps_smoke";

# We export the directory names
export FLOWCYTO_RAW_DIR="${FLOWCYTO_ROOT}/raw_data";
export FLOWCYTO_OUT_DATA="${FLOWCYTO_ROOT}/out/data"
export FLOWCYTO_RUNS_DIR="${FLOWCYTO_ROOT}/out/dumps/runs";
export FLOWCYTO_RUNS_SMOKE_DIR="${FLOWCYTO_ROOT}/out/dumps/runs_smoke";
export FLOWCYTO_SWEEPS_DIR="${FLOWCYTO_ROOT}/out/dumps/sweeps";
export FLOWCYTO_SWEEPS_SMOKE_DIR="${FLOWCYTO_ROOT}/out/dumps/sweeps_smoke";

# We export the variable for the raw data paths.
export FLOWCYTO_OLD_DATA_RAW="${FLOWCYTO_RAW_DIR}/OH_OH0013_OLD_UMAP_annotated_red.h5ad"
export FLOWCYTO_NEW_DATA_RAW="${FLOWCYTO_RAW_DIR}/260331_OH013_New_Panel_annotated_red.h5ad"

# We also export the variables for the new data paths
export FLOWCYTO_OLD_DATA="${FLOWCYTO_OUT_DATA}/OH_OH0013_OLD_UMAP_annotated_red.zarr"
export FLOWCYTO_NEW_DATA="${FLOWCYTO_OUT_DATA}/260331_OH013_New_Panel_annotated_red.zarr"
```

## 1. Download the data from the drive.

Then, make sure that the `download_gdrive.py` file is present with

```bash
ls "${FLOWCYTO_ROOT}";
```

Run it with:

```bash
cd "${FLOWCYTO_ROOT}" && \
  python "${FLOWCYTO_ROOT}"/download_gdrive.py \
    --output "${FLOWCYTO_RAW_DIR}" \
    2>&1 | tee download_gdrive.log
```

This will dowload three files in the from the Google drive -- written by default to the `"${FLOWCYTO_ROOT}/raw_data/"` directory -- containing data from three experiments:

1. **Spectral flow cytometry measurements with the old panel**: `OH_OH0013_OLD_UMAP_annotated_red.h5ad`.
2. **Spectral flow cytometry measurements with the new panel**: `260331_OH013_New_Panel_annotated_red.h5ad`.
3. **CITESeq measurements**: `/lustre/groups/ml01/projects/FlowCytometry/raw_data/260413_HID_SAK_NewAnno_15k_v4.2_final_redmod.h5ad`.

Let's double check the contents of the raw data directory:

```bash
ls "${FLOWCYTO_RAW_DIR}";
# Expected: -> 260331_OH013_New_Panel_annotated_red.h5ad          OH_OH0013_OLD_UMAP_annotated_red.h5ad 260413_HID_SAK_NewAnno_15k_v4.2_final_redmod.h5ad
```

---

## 2. Prepare the data.

First, download `uv` with:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Then, clone the `theislab/flowrun` repository:

```bash
cd ~/repos
git clone git@github.com:selmanozleyen/flowrun.git;
cd flowrun;
```

Create `uv` environment and activate it:

```bash
uv venv
source .venv/bin/activate
```

Update the repository and synchronize the environment:

```bash
git fetch && git pull && uv sync --locked --extra sweep;
```

Then run the script to prepare the data with `uv`:

```bash
uv run python -m flowrun.panels \
  --out "${FLOWCYTO_OUT_DATA}" \
  --rename CD41a=CD41 EPCR=CD201 \
  --panel "old panel" "${FLOWCYTO_OLD_DATA_RAW}" \
  --panel "new panel" "${FLOWCYTO_NEW_DATA_RAW}" \
  2>&1 | tee prepare_data.log
```

The new data will live in `"${FLOWCYTO_ROOT}/out/data"`, let's double-check its contents:

```bash
ls "${FLOWCYTO_ROOT}/out/data"
# --> Expected output: 260331_OH013_New_Panel_annotated_red.zarr  OH_OH0013_OLD_UMAP_annotated_red.zarr
```

---

## 3. Training the model

### 3.1 Resolve and inspect run's configurations.

Resolve the `hydra` configuration and inspect them:

```bash
uv run python -m flowrun.fit_cfm \
    +experiment=flow_cyto_panel_alignment \
    paths.target_adata="['${FLOWCYTO_OLD_DATA}','${FLOWCYTO_NEW_DATA}']" \
    --cfg job --resolve \
    2>&1 | tee resolved_config.log;
```

Then inspect the output with:

```bash
tail -n 10 resolved_config.log;
# --> Expected output:
#   ckpt_path: null
# run:
#   loader_seed: 42
#   splitter_seed: 42
#   method_seed: 0
#   allow_overwrite: true
# panels:
#   layer: logicle_trans
# save:
#   run_dir: ./runs/cfm_run
```

Clean up the log file right after, to avoid clutter:

```bash
rm resolved_config.log;
```

### 3.2 Execute smoke run.

```bash
RUN_DIR="${FLOWCYTO_RUNS_SMOKE_DIR}/$(date +%Y-%m-%d_%H-%M-%S)"
uv run python -m flowrun.fit_cfm \
    +experiment=flow_cyto_panel_alignment_smoke \
    paths.target_adata="['${FLOWCYTO_OLD_DATA}','${FLOWCYTO_NEW_DATA}']" \
    save.run_dir="${RUN_DIR}" \
    callbacks.model_checkpoint.dirpath="${RUN_DIR}/checkpoints" \
    loggers.wandb.save_dir="${RUN_DIR}/wandb" \
    loggers.csv.save_dir="${RUN_DIR}/csv" \
    trainer.default_root_dir="${RUN_DIR}/logs" \
    trainer.accelerator=cpu \
    2>&1 | tee train_smoke.log
```

### 3.3 Run the actual job.

```bash
RUN_DIR="${FLOWCYTO_RUNS_DIR}/$(date +%Y-%m-%d_%H-%M-%S)"
uv run python -m flowrun.fit_cfm \
    +experiment=flow_cyto_panel_alignment \
    paths.target_adata="['${FLOWCYTO_OLD_DATA}','${FLOWCYTO_NEW_DATA}']" \
    save.run_dir="${RUN_DIR}" \
    callbacks.model_checkpoint.dirpath="${RUN_DIR}/checkpoints" \
    loggers.wandb.save_dir="${RUN_DIR}/wandb" \
    loggers.wandb.offline=False \
    loggers.csv.save_dir="${RUN_DIR}/csv" \
    trainer.default_root_dir="${RUN_DIR}/logs" \
    2>&1 | tee train.log
```
---

## 4. Running the hyper-parameters sweep.

We use in this example the FGW OT, but this would be the same for the linear case.

### 4.1 Resolve and inspect the run's configurations.

```bash
uv run --extra sweep python -m flowrun.fit_cfm \
    +experiment=flow_cyto_panel_alignment_sweep_fused \
    paths.target_adata="['${FLOWCYTO_OLD_DATA}','${FLOWCYTO_NEW_DATA}']" \
    --cfg job --resolve \
    2>&1 | tee resolved_config_sweep_fused.log;
```

Then inspect the output with:

```bash
tail -n 10 resolved_config_sweep_fused.log;
# --> Expected output:
# run:
#   loader_seed: 42
#   splitter_seed: 42
#   method_seed: 0
#   allow_overwrite: false
# panels:
#   layer: logicle_trans
# save:
#   run_dir: None/save
# sweep_name: flow_cyto_panel_alignment_sweep_fused
```

Clean up the log file right after, to avoid clutter:

```bash
rm resolved_config_sweep_fused.log;
```

### 4.2 Execute smoke run.

First, we need to set the environmental variable for hydra logging; this will be referenced across the configurations to save everything in one place.

* NOTE: The `--multirun` flag is needed to submit jobs in series.

```bash
export FLOWRUN_SWEEP_DIR="${FLOWCYTO_SWEEPS_SMOKE_DIR}";
uv run --extra sweep python -m flowrun.fit_cfm --multirun \
    +experiment=flow_cyto_panel_alignment_sweep_smoke \
    paths.target_adata="['${FLOWCYTO_OLD_DATA}','${FLOWCYTO_NEW_DATA}']" \
    trainer.accelerator=cpu \
    2>&1 | tee sweep_fused_smoke.log
```

### 4.3 Run actual sweep.

```bash
export FLOWRUN_SWEEP_DIR="${FLOWCYTO_SWEEPS_DIR}";
uv run --extra sweep python -m flowrun.fit_cfm --multirun \
    +experiment=flow_cyto_panel_alignment_sweep_fused \
    paths.target_adata="['${FLOWCYTO_OLD_DATA}','${FLOWCYTO_NEW_DATA}']" \
    loggers.wandb.offline=False \
    hydra/launcher=submitit_slurm \
    hydra.launcher.partition="gpu_p" \
    hydra.launcher.qos="gpu_normal" \
    hydra.launcher.cpus_per_task=6 \
    hydra.launcher.mem_gb=64 \
    hydra.launcher.gres="gpu:1" \
    hydra.launcher.timeout_min=1440 \
    hydra.launcher.array_parallelism=10 \
    2>&1 | tee sweep_fused.log
```

### 4.4 Submit sweep as an actual job.

```bash
export FLOWRUN_SWEEP_DIR="${FLOWCYTO_SWEEPS_DIR}";
SWEEP_TYPE="flow_cyto_panel_alignment_sweep_linear_massive_model";
sbatch \
  --job-name="${SWEEP_TYPE}" \
  --partition=cpu_p \
  --cpus-per-task=1 \
  --qos=cpu_normal \
  --mem=8G \
  --time=24:00:00 \
  --chdir="${HOME}/repos/flowrun" \
  --output="${FLOWCYTO_ROOT}/out/dumps/logs/${SWEEP_TYPE}-%j.out" \
  --wrap="source .venv/bin/activate && \
    export FLOWRUN_SWEEP_DIR=\"${FLOWRUN_SWEEP_DIR}\" && \
    uv run --with hydra-submitit-launcher --extra sweep python -m flowrun.fit_cfm --multirun \
      +experiment="${SWEEP_TYPE}" \
      hydra/launcher=submitit_slurm \
      hydra.sweeper.n_jobs=10 \
      paths.target_adata=\"['${FLOWCYTO_OLD_DATA}','${FLOWCYTO_NEW_DATA}']\" \
      loggers.wandb.offline=False \
      hydra.launcher.partition=\"gpu_p\" \
      hydra.launcher.qos=\"gpu_normal\" \
      hydra.launcher.cpus_per_task=6 \
      hydra.launcher.mem_gb=64 \
      hydra.launcher.gres=\"gpu:1\" \
      hydra.launcher.timeout_min=1440 \
      hydra.launcher.array_parallelism=10 \
      trainer.max_epochs=100 \
      callbacks.early_stopping.patience=30 \
      2>&1 | tee sweep_sbatch.log"
```

---

## *Comments*

Maybe we can consider adding the following block to the configuration of the sweeps, in order to submit the sweep as an array of jobs:

```yaml
hydra:
  launcher:
    partition: gpu_p
    qos: gpu_normal
    cpus_per_task: 6
    mem_gb: 64
    gres: gpu:1
    timeout_min: 1440
    array_parallelism: 10
```

For the moment, these fields are overriden through CLI in the submission of the sweep at point 4.3 and 4.4.

---

## TODOs and next steps:

- [] Check that the runs went through and how they converged. 
- [] Write script for prediction.

- [] Increase grid:
    - [] Bigger models.
    - [] More coupling configurations.
    - [] More hyperparamaters:
        - [] Maybe do as as Selman suggested: start generation from uniform over hypercube instead of isotropic gaussian; this would require changes to `sckitflow`.
        - [] Maybe try different time features and conditioning: we should anyway implement the `FiLM` conditionining in `sckitflow.`
    - [] Different paths?
