# flowrun

Pipeline steps around [sckitflow](https://github.com/theislab/sckitflow): split and PCA steps and a
Hydra entry point that trains a flow-matching model from one YAML.

## Install

```bash
uv sync --extra test
```

## Train

```bash
uv run flowrun-fit-cfm paths.target_adata=/path/to/data.zarr save.run_dir=runs/first
```

Every key and its default lives in `src/flowrun/configs/fit_cfm.yaml`. The run is saved with
`sckitflow.Model.save_pretrained`, so `sckitflow.Model.from_pretrained(run_dir, adata)` reloads it.

## CellFlow

```bash
uv run flowrun-fit-cfm +experiment=cellflow paths.target_adata=/path/to/data.zarr save.run_dir=runs/cellflow
```

`paths.target_adata` also takes a list, read as one dataset (e.g. `[ctrl.zarr,pert.zarr]`); each file gets
its own split column and PCA rows written back.

`configs/experiment/cellflow.yaml` holds CellFlow's settings (OT-matched CFM on `X_pca`, drug combinations
pooled with attention, cell line as the group); `cellflow_smoke` is a small version for checking the wiring.
`genot` layers GENOT on top (generate from noise, conditioned on the encoded source cell):
`+experiment=[cellflow,genot]`.

## Panel imputation and sweeps

`+experiment=flow_cyto_panel_alignment` maps cells from an old flow cytometry marker panel onto a new one (fused Gromov-Wasserstein,
a source encoder); `+experiment=flow_cyto_panel_alignment_sweep_fused` and `flow_cyto_panel_alignment_sweep_linear` sweep it with fused and linear OT
with Optuna on SLURM (`uv sync --extra sweep`, then `flowrun-fit-cfm -m +experiment=flow_cyto_panel_alignment_sweep_fused`). The runbook is `docs/sweeps.md`.

## Layout

- `flowrun.split`: `train` / `val` / `test` labels for targets in an obs column, and `all` for the sources every
  split shares. `fit_cfm` writes the split into the input file's `obs` and reuses it on later runs
  (`splitter.overwrite=true` re-splits).
  `shared_source_combination` holds out condition combinations, `shared_source_holdout` holds out chosen
  values of one column (e.g. cell lines).
- `flowrun.preprocess`: PCA (`scanpy.pp.pca` on dask), z-score and whitening, fitted on the training split
  (reading only those rows' chunks). Every cell's output goes to `obsm[target_sample_rep]` and the fitted steps to
  `uns["preprocessing"]` in the input file; `apply` repeats them on new cells, `inverse` maps predictions back.
- `flowrun.fit_cfm`: the training entry point.
