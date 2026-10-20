# flowrun

Pipeline steps around [sckitflow](https://github.com/theislab/sckitflow): data transforms and a
Hydra entry point that trains a flow-matching model from one YAML.

## Install

```bash
uv sync --extra test
```

## Train

```bash
uv run flowrun-fit-cfm paths.target_adata=/path/to/data.h5ad save.run_dir=runs/first
```

Every key and its default lives in `src/flowrun/configs/fit_cfm.yaml`. The run is saved with
`sckitflow.Model.save_pretrained`, so `sckitflow.Model.from_pretrained(run_dir, adata)` reloads it.

## Layout

- `flowrun.preprocessing.transforms`: PCA, z-score and whitening, fitted on the training split and
  saved next to the run.
- `flowrun.fit_cfm`: the training entry point.
