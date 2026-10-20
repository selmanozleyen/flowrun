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

`configs/experiment/cellflow.yaml` holds CellFlow's settings (OT-matched CFM on `X_pca`, drug combinations
pooled with attention, cell line as the group); `cellflow_smoke` is a small version for checking the wiring.

## Layout

- `flowrun.split`: `train` / `val` / `test` labels for targets in an obs column, and `all` for the sources every
  split shares;
  `shared_source_combination` holds out condition combinations, `shared_source_holdout` holds out chosen
  values of one column (e.g. cell lines).
- `flowrun.pca`: `scanpy.pp.pca` on dask, fitted on the training split (reading only those rows' chunks); every cell's projection and the
  basis are written into the input file (`obsm[sample_rep]`, `varm["PCs"]`, `uns["pca"]`).
- `flowrun.fit_cfm`: the training entry point.
