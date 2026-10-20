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

## Layout

- `flowrun.split`: `train` / `val` / `test` labels for an obs column, with sources shared across splits;
  `shared_source_combination` holds out condition combinations, `shared_source_holdout` holds out chosen
  values of one column (e.g. cell lines).
- `flowrun.pca`: PCA fitted on the training split, reading only those rows; the basis is stored in the AnnData
  and saved to `run_dir/pca.zarr` (obs/var names map it to the original cells and genes).
- `flowrun.fit_cfm`: the training entry point.
