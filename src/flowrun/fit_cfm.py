"""Full training pipeline for a Conditional Flow Matching (CFM) model.

Wires together the components needed to train a Conditional Flow Matching
model on single-cell data with an MLP-parameterised velocity field. Designed
to be invoked via Hydra as the entry point of a training run:

    flowrun-fit-cfm [+config overrides]

Pipeline (in the order ``main`` executes them)
----------------------------------------------
1. Seed everything from ``config.run.method_seed`` (``lightning.seed_everything``).
2. Load AnnData from disk (``read_adatas``; paths in ``config.paths``).
3. Write the split column from ``config.splitter`` (``flowrun.split``: control/train/val/test
   labels) and a ``FlowDataModule`` from ``config.datamanager`` + ``config.datamodule``.
4. Build the MLP velocity field (``MLPVelocityConfig`` from ``config.module``).
5. Build the base training method (``CFMTrainingConfig``), wrap it in a
   ``MatchedTrainingConfig`` that pairs source/target samples according to
   ``config.coupling``, and build the inference method (``ODEInferenceConfig``).
   Probability path and samplers arrive as portable ``*Config`` objects so
   ``Model.save_pretrained`` can serialize them; each method resolves them to a callable
   inside ``build``.
6. Build a ``torch.optim`` optimizer and validation metrics
   (``EnergyDistance``, ``MaximumMeanDiscrepancy``) from ``config.metrics``.
7. Build a ``TrainingPlan`` (the ``LightningModule`` wrapper) from
   ``config.training_plan``, wiring together the training/inference methods,
   metrics and optimizer.
8. Configure ``WandbLogger`` + ``CSVLogger``, callbacks (``EarlyStopping``,
   ``GradientAccumulationScheduler``, ``LearningRateMonitor``, ``ModelCheckpoint``),
   and the ``lightning.pytorch.Trainer``.
9. ``trainer.fit(...)``.
10. Persist the model via ``Model.save_pretrained`` (writes ``config.json`` +
    ``module/model.safetensors``). Training checkpoints are written separately by
    the ``ModelCheckpoint`` callback during ``trainer.fit``. How the run went (optimizer, seeds, batch size) is this config's, not
    the model's.
11. Return the metric ``config.callbacks.model_checkpoint.monitor`` names (e.g. ``"val/edist"``), for sweeps:
    the checkpoint's best score, else the last logged value, else ``nan`` (a sweep marks that trial failed).

Key concepts
------------
- **Conditional Flow Matching (CFM)**: learns a velocity field v(x, t | c) that
  transports a source distribution to a target distribution conditioned on c.
- **Probability path**: defines the interpolation between source and target
  samples. Supported: ``linear_dirac``, ``linear_gaussian``,
  ``schrodinger_bridge``, ``variance_preserving``. See
  ``sckitflow.core.probability_paths``.
- **Velocity field**: the network that predicts the time derivative along the
  path. Here an MLP with separate state/time/condition encoders.
- **Coupling**: at training time, pairs source samples with target samples
  before regression. Selected by ``config.coupling.method``: ``independent``,
  ``exact``, ``sinkhorn``, ``partial``, ``unbalanced``,
  ``gromov-wasserstein``, ``fused``. See ``sckitflow.core.coupling``.
- **Sampler**: time and noise draws during training / inference.
  ``time_sampler`` and ``noise_sampler`` in the config are ``SamplerConfig``
  objects (portable — the run's spec serializes them). See
  ``sckitflow.core._samplers``.
- **PCA**: optional, fitted on the training split only and written to
  ``obsm[target_sample_rep]``, with z-score and whitening. Configured via ``config.preprocessing``; see
  ``flowrun.preprocess``.
- **Inference**: integrates the learned ODE from t=0 to t=1 with ``torchdiffeq``,
  configured via ``ODEInferenceConfig``.

Configuration
-------------
The full YAML is ``flowrun/configs/fit_cfm.yaml`` (Hydra).
See that file for every key, its meaning, and its default. The coupling
selection and its hyperparameters live under ``config.coupling``.
``hydra.job.chdir: false`` is recommended so relative paths
(``save.run_dir``, ``callbacks.model_checkpoint.dirpath``, logger
``save_dir``) resolve from the launch directory rather than Hydra's
``outputs/<date>/<time>/`` tree.

Outputs
-------
- Checkpoints: ``config.callbacks.model_checkpoint.dirpath`` (written by the
  ``ModelCheckpoint`` callback during training; best-``val_edist`` weights
  plus ``last.ckpt``).
- Loggers: ``config.loggers.{wandb,csv}.save_dir``.
- Saved model: ``config.save.run_dir`` (via ``Model.save_pretrained``; contains
  ``config.json`` — the portable ``ModelConfig`` — and ``module/model.safetensors``).
  Reload it with ``sckitflow.Model.from_pretrained(run_dir, adata)`` for
  **inference**.
- Return value: the best (else last) value of the monitored metric, ``nan`` if it was never logged.
"""

from pathlib import Path
from typing import Any

import anndata as ad
import hydra
import numpy as np
import pandas as pd
import torch
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import (
    EarlyStopping,
    GradientAccumulationScheduler,
    LearningRateMonitor,
    ModelCheckpoint,
)
from lightning.pytorch.loggers import CSVLogger, WandbLogger
from omegaconf import DictConfig, ListConfig, OmegaConf
from sckitflow import (
    IsotropicGaussianNoiseSamplerConfig,
    Model,
    ModelConfig,
    UniformTimeSamplerConfig,
    get_logger,
)
from sckitflow.core.coupling import (
    ExactCouplingConfig,
    FusedGromovWassersteinCouplingConfig,
    GromovWassersteinCouplingConfig,
    IndependentCouplingConfig,
    PartialCouplingConfig,
    SinkhornCouplingConfig,
    UnbalancedCouplingConfig,
)
from sckitflow.core.methods import CFMTrainingConfig, MatchedTrainingConfig, ODEInferenceConfig
from sckitflow.core.metrics import EnergyDistance, MaximumMeanDiscrepancy
from sckitflow.core.nn import MLPVelocityConfig
from sckitflow.core.probability_paths import (
    LinearDiracProbabilityPathConfig,
    LinearGaussianProbabilityPathConfig,
    SchrodingerBridgeProbabilityPathConfig,
    VariancePreservingDiracProbabilityPathConfig,
)
from sckitflow.data import DataManagerConfig, FlowDataModule
from sckitflow.trainer import TrainingPlan

from flowrun import split
from flowrun._store import write_obs_column
from flowrun.preprocess import preprocess, write_preprocessing

logger = get_logger(__name__)

EDIST_KEY = "edist"
MMD_KEY = "mmd"


def resolve_omegaconf_to_dictionary(conf_dict: dict[str, Any] | None | DictConfig) -> dict[str, Any]:
    """Resolves the omegaconf ``DictConfig`` to a plain dictionary."""
    out_dict = {}
    if conf_dict is not None:
        out_dict = conf_dict
        if isinstance(out_dict, dict):
            out_dict = OmegaConf.create(out_dict)
        out_dict = OmegaConf.to_container(out_dict, resolve=True)
    return out_dict


def enable_zarrs() -> None:
    """Use the ``zarrs`` (Rust) codec pipeline for all zarr I/O, and fail rather than fall back to the slow one."""
    import zarr
    from zarr.registry import get_pipeline_class

    zarr.config.set({"codec_pipeline.path": "zarrs.ZarrsCodecPipeline"})
    cls = get_pipeline_class()
    if f"{cls.__module__}.{cls.__name__}" != "zarrs.ZarrsCodecPipeline":
        raise RuntimeError(f"zarrs codec pipeline did not load (active: {cls.__module__}.{cls.__name__}).")


def as_list(value) -> list:
    """A single config value, or a list of them, as a list."""
    return list(value) if isinstance(value, ListConfig | list | tuple) else [value]


def read_zarr_input(path: Path, config: DictConfig) -> ad.AnnData:
    """Read only what the run uses from a zarr AnnData: the obs columns and uns tables the schema and the splitter
    name, with ``X`` and every ``obsm`` rep left on disk (backed, so unread ones cost nothing)."""
    import zarr
    from scfit.data._io import load_backed_adata

    dm, splitter = config.datamanager, config.splitter
    cols = {
        *dm.groups,
        *(c for level in dm.categorical_conditions.values() for c in level),
        dm.split_by,
        splitter.source_key,
        splitter.key,
        *splitter.group_keys,
    }
    uns = {*dm.categorical_reps.values(), *dm.groups_reps.values()}
    group = zarr.open_group(path, mode="r")
    keys = ["X", *(f"obsm/{k}" for k in group.get("obsm", {}))]
    return load_backed_adata(
        group,
        keys=keys,
        cols=sorted(c for c in cols if c is not None and c in group["obs"]),
        uns=sorted(k for k in uns if k in group.get("uns", {})),
    )


def split_labels(obs, splitter: DictConfig, *, seed: int):
    """The split column ``config.splitter`` describes."""
    is_source = (
        obs[splitter.source_key].astype(str).to_numpy() == str(splitter.source_value)
        if splitter.source_key is not None
        else np.zeros(len(obs), dtype=bool)
    )
    if splitter.type == "shared_source_combination":
        return split.shared_source_combination(
            obs,
            splitter.group_keys,
            is_source=is_source,
            always_train_keys=splitter.always_train_keys,
            test_fraction=splitter.test_fraction,
            val_fraction=splitter.val_fraction,
            seed=seed,
        )
    if splitter.type == "shared_source_holdout":
        return split.shared_source_holdout(obs, splitter.key, is_source=is_source, test=splitter.test, val=splitter.val)
    raise ValueError(
        f"Unknown splitter.type {splitter.type!r}; use 'shared_source_combination' or 'shared_source_holdout'."
    )


def val_from_train(adatas: list[ad.AnnData], splitter: DictConfig, *, train: str, seed: int) -> None:
    """Moves ``splitter.val_from_train`` of each group's ``train`` rows to ``val``, in memory only, so a split
    reused from files without a validation split gets one and the files keep theirs."""
    rng = np.random.default_rng(seed)
    for a in adatas:
        labels = a.obs[splitter.split_key].astype(str).to_numpy().copy()
        groups = a.obs[list(splitter.group_keys)].astype(str).agg("|".join, axis=1).to_numpy()
        for g in np.unique(groups):
            rows = np.flatnonzero((groups == g) & (labels == train))
            labels[rng.choice(rows, size=round(len(rows) * splitter.val_from_train), replace=False)] = "val"
        a.obs[splitter.split_key] = pd.Categorical(labels)


@hydra.main(config_path="configs", config_name="fit_cfm", version_base=None)
def main(config: DictConfig) -> float:
    """Main entrypoint for CFM training."""
    # Seed everything with lightning
    # This is the master seeding, and it will be also saved to the run
    # for further reproducibility.
    enable_zarrs()
    logger.info(f"Setting the global run seed to {config.run.method_seed}")
    seed_everything(seed=config.run.method_seed, workers=True)

    # Load data from disk using the utility function.
    # Please override this function to allow custom operations on the
    # data after loading; at the moment, only vanilla loading
    # with ``anndata`` is performed, and no other transformation is
    # applied to the data.
    # One file or a list read as one dataset (rows in list order), each file with only what the run uses.
    paths = [Path(p) for p in as_list(config.paths.target_adata)]
    adatas = [read_zarr_input(p, config) if p.suffix == ".zarr" else ad.read_h5ad(p) for p in paths]
    logger.info(
        f"Read {len(adatas)} file(s): " + ", ".join(f"{p.name} ({a.n_obs} cells)" for p, a in zip(paths, adatas))
    )

    # Split: one label per row in ``obs[splitter.split_key]``, read back as ``split_by``. A split already in the
    # input files is reused, so every model trained on them shares it.
    splitter = config.splitter
    without = [p.name for p, a in zip(paths, adatas, strict=True) if splitter.split_key not in a.obs.columns]
    if without and len(without) < len(paths) and not splitter.overwrite:
        raise ValueError(
            f"obs[{splitter.split_key!r}] is in some input files but not in {without}, and one split must cover "
            "every file. Set splitter.overwrite=true to split them all again."
        )
    new_split = bool(without) or splitter.overwrite
    if new_split:
        obs = pd.concat([a.obs for a in adatas], ignore_index=True)
        labels = split_labels(obs, splitter, seed=config.run.splitter_seed)
        for a, part in zip(adatas, np.split(labels.to_numpy(), np.cumsum([a.n_obs for a in adatas])[:-1]), strict=True):
            a.obs[splitter.split_key] = pd.Categorical(part, categories=labels.cat.categories)
    else:
        logger.info(f"Reusing the split in obs[{splitter.split_key!r}]; set splitter.overwrite=true to re-split.")
        if splitter.val_from_train and not any((a.obs[splitter.split_key] == "val").any() for a in adatas):
            logger.info(
                f"The split has no val: moving {splitter.val_from_train:.0%} of each group's training rows to val."
            )
            val_from_train(adatas, splitter, train=config.datamodule.train_split, seed=config.run.splitter_seed)

    # Preprocessing is fitted on the training targets and the shared sources, so held-out cells never shape it.
    pp = config.preprocessing
    preprocessed = pp.pca is not None or pp.zscore or pp.whitening is not None
    if preprocessed:
        if pp.pca is None and pp.input_key == config.datamanager.target_sample_rep:
            raise ValueError(
                f"preprocessing.input_key and datamanager.target_sample_rep are both {pp.input_key!r}: the preprocessing "
                "would overwrite its own input. Write it to another key (datamanager.target_sample_rep), which the model "
                "then reads."
            )
        train_mask = np.concatenate([a.obs[splitter.split_key].isin(["train", "all"]).to_numpy() for a in adatas])
        preprocess(
            adatas,
            train_mask=train_mask,
            pca=pp.pca,
            n_hvg=pp.n_hvg,
            hvg_flavor=pp.hvg_flavor,
            zscore=pp.zscore,
            whitening=pp.whitening,
            epsilon=pp.epsilon,
            input_key=pp.input_key,
            key_added=config.datamanager.target_sample_rep,
        )

    # Prepare data manager.
    # The ``DataManager`` is never materialized directly here: it lives
    # in the ``datamodule``, of which it is an attribute.
    logger.info("Initializing datamanager.")
    # Plain containers: an OmegaConf node inside the spec cannot be saved with the model.
    dm_cfg = DataManagerConfig(**OmegaConf.to_container(config.datamanager, resolve=True))

    # Initialize datamodule.
    # The loaders' sampling schedule is seeded from ``config.run.loader_seed`` for reproducibility.
    logger.info(f"Initializing datamodule with seed {config.run.loader_seed}")
    datamodule_rng = np.random.default_rng(config.run.loader_seed)
    datamodule = FlowDataModule.from_adata(
        adatas if len(adatas) > 1 else adatas[0],
        dm_cfg.build(),
        train_split=config.datamodule.train_split,
        n_train_steps=config.datamodule.n_train_steps,
        batch_size=config.datamodule.batch_size,
        dtype=getattr(torch, config.datamodule.dtype),
        loader_kwargs={
            **resolve_omegaconf_to_dictionary(config.datamodule.loader_kwargs),
            "seed": int(datamodule_rng.integers(2**63)),
        },
    )

    # Initialize neural network.
    # The dimensionalities are automatically inferred from the
    # data_dims object, linked to the datamodule.
    logger.info("Initializing neural velocity field.")
    module_cfg = MLPVelocityConfig(
        encode_state=config.module.encode_state,
        encode_time=config.module.encode_time,
        time_features_id=config.module.time_features_id,
        num_time_features=config.module.num_time_features,
        max_period=config.module.max_period,
        state_encoder_output_dim=config.module.state_encoder_output_dim,
        time_encoder_output_dim=config.module.time_encoder_output_dim,
        state_encoder_mlp_kwargs=resolve_omegaconf_to_dictionary(config.module.state_encoder_mlp_kwargs),
        time_encoder_mlp_kwargs=resolve_omegaconf_to_dictionary(config.module.time_encoder_mlp_kwargs),
        vf_decoder_mlp_kwargs=resolve_omegaconf_to_dictionary(config.module.vf_decoder_mlp_kwargs),
        conditioning_id=config.module.conditioning_id,
        conditioning_kwargs=resolve_omegaconf_to_dictionary(config.module.conditioning_kwargs),
        condition_encoder_input_layers=resolve_omegaconf_to_dictionary(config.module.condition_encoder_input_layers)
        or None,
        condition_encoder_output_dim=config.module.condition_encoder_output_dim,
        condition_encoder_pooling_mode=config.module.condition_encoder_pooling_mode,
        condition_encoder_pooling_kwargs=resolve_omegaconf_to_dictionary(
            config.module.condition_encoder_pooling_kwargs
        ),
        condition_encoder_pooling_proj_dim=config.module.condition_encoder_pooling_proj_dim,
        condition_encoder_pooling_proj_bias=config.module.condition_encoder_pooling_proj_bias,
        condition_encoder_output_layers_kwargs=resolve_omegaconf_to_dictionary(
            config.module.condition_encoder_output_layers_kwargs
        ),
        source_encoder_mlp_kwargs=None
        if config.module.source_encoder_mlp_kwargs is None
        else resolve_omegaconf_to_dictionary(config.module.source_encoder_mlp_kwargs),
        source_encoder_output_dim=config.module.source_encoder_output_dim,
    )
    module = module_cfg.build(datamodule.data_dims)

    # Initialize probability path
    logger.info("Initializing probability path.")
    if config.probability_path.name == "linear_dirac":
        probability_path_cfg = LinearDiracProbabilityPathConfig(sigma=config.probability_path.sigma)
    elif config.probability_path.name == "linear_gaussian":
        probability_path_cfg = LinearGaussianProbabilityPathConfig(sigma=config.probability_path.sigma)
    elif config.probability_path.name == "schrodinger_bridge":
        probability_path_cfg = SchrodingerBridgeProbabilityPathConfig(
            sigma=config.probability_path.sigma, eps=config.probability_path.eps
        )
    elif config.probability_path.name == "variance_preserving":
        probability_path_cfg = VariancePreservingDiracProbabilityPathConfig(sigma=config.probability_path.sigma)
    else:
        raise ValueError(f"Invalid probability path name {config.probability_path.name}.")

    # Initialize time sampler
    logger.info("Initializing time sampler.")
    if config.time_sampler.name == "uniform":
        time_sampler_cfg = UniformTimeSamplerConfig()
    else:
        raise ValueError(f"Invalid time sampler name {config.time_sampler.name}.")

    # Initialize noise sampler
    logger.info("Initializing noise sampler.")
    if config.noise_sampler.name == "isotropic_gaussian":
        noise_sampler_cfg = IsotropicGaussianNoiseSamplerConfig()
    else:
        raise ValueError(f"Invalid noise sampler name {config.noise_sampler.name}.")

    # Initialize training module.
    logger.info("Initializing base training method.")
    base_training_cfg = CFMTrainingConfig(
        probability_path=probability_path_cfg,
        time_sampler=time_sampler_cfg,
        noise_sampler=noise_sampler_cfg,
        generate_from_noise=config.training.generate_from_noise,
    )

    # Initialize coupling configuration and matched training configurations.
    logger.info("Initializing coupling module.")
    if config.coupling.method == "independent":
        coupling_cfg = IndependentCouplingConfig()
    elif config.coupling.method == "exact":
        coupling_cfg = ExactCouplingConfig(
            scale_cost=config.coupling.scale_cost,
            num_iter_max=config.coupling.num_iter_max,
        )
    elif config.coupling.method == "sinkhorn":
        coupling_cfg = SinkhornCouplingConfig(
            scale_cost=config.coupling.scale_cost,
            num_iter_max=config.coupling.num_iter_max,
            reg=config.coupling.reg,
            stop_thr=config.coupling.stop_thr,
        )
    elif config.coupling.method == "partial":
        coupling_cfg = PartialCouplingConfig(
            scale_cost=config.coupling.scale_cost,
            num_iter_max=config.coupling.num_iter_max,
            reg=config.coupling.reg,
            stop_thr=config.coupling.stop_thr,
            m=config.coupling.m,
        )
    elif config.coupling.method == "unbalanced":
        coupling_cfg = UnbalancedCouplingConfig(
            scale_cost=config.coupling.scale_cost,
            num_iter_max=config.coupling.num_iter_max,
            reg=config.coupling.reg,
            stop_thr=config.coupling.stop_thr,
            reg_m=config.coupling.reg_m,
        )
    elif config.coupling.method == "gromov-wasserstein":
        coupling_cfg = GromovWassersteinCouplingConfig(
            scale_cost=config.coupling.scale_cost,
            epsilon=config.coupling.epsilon,
            max_iter=config.coupling.max_iter,
            tol=config.coupling.tol,
        )
    elif config.coupling.method == "fused":
        coupling_cfg = FusedGromovWassersteinCouplingConfig(
            scale_cost=config.coupling.scale_cost,
            epsilon=config.coupling.epsilon,
            max_iter=config.coupling.max_iter,
            tol=config.coupling.tol,
            alpha=config.coupling.alpha,
        )
    else:
        raise ValueError(f"Invalid coupling method {config.coupling.method}")

    # Initialize wrapped method for matched training.
    logger.info("Initializing matched wrapper.")
    training_cfg = MatchedTrainingConfig(
        method=base_training_cfg,
        coupling=coupling_cfg,
    )

    logger.info("Building training method.")
    training_method = training_cfg.build(
        module,
        dims=datamodule.data_dims,
    )

    # Initialize inference module.
    logger.info("Initializing inference method.")
    inference_cfg = ODEInferenceConfig(
        probability_path=probability_path_cfg,
        time_sampler=time_sampler_cfg,
        noise_sampler=noise_sampler_cfg,
        generate_from_noise=config.inference.generate_from_noise,
        solver_kwargs=resolve_omegaconf_to_dictionary(config.inference.solver_kwargs),
        return_trajectory=config.inference.return_trajectory,
        n_steps=config.inference.n_steps,
        n_samples=config.inference.n_samples,
    )
    logger.info("Building inference method.")
    inference_method = inference_cfg.build(module, dims=datamodule.data_dims)

    # Prepare optimizer and scheduler.
    logger.info("Initializing optimizer.")
    optimizer = getattr(torch.optim, config.optimizer.name)(
        module.parameters(), **resolve_omegaconf_to_dictionary(config.optimizer.kwargs)
    )
    lr_scheduler = None

    # Initialize metrics used to evaluate model's predictions during validation.
    logger.info("Initializing metrics.")
    edist = EnergyDistance(
        dist_sync_on_step=config.metrics.edist.dist_sync_on_step, max_samples=config.metrics.edist.max_samples
    )
    mmd = MaximumMeanDiscrepancy(
        gammas=config.metrics.mmd.gammas,
        dist_sync_on_step=config.metrics.mmd.dist_sync_on_step,
    )
    metrics = {
        EDIST_KEY: edist,
        MMD_KEY: mmd,
    }

    # Initialize prediction transformations.
    # TODO: wire these to ``config.transforms.{pred,target}`` when that section
    # exists; leaving them ``None`` runs the plan without post-hoc transforms.
    pred_transform = None  # TODO

    # Initialize target transformations.
    target_transform = None  # TODO

    # Initialize training plan.
    logger.info("Initializing training plan.")
    plan = TrainingPlan(
        training_method,
        optimizer,
        lr_scheduler=lr_scheduler,
        lr_scheduler_interval=config.training_plan.lr_scheduler_interval,
        inference_method=inference_method,
        metrics=metrics,
        val_names=datamodule.val_names,
        pred_transform=pred_transform,
        target_transform=target_transform,
        seed=config.run.method_seed,
    )

    # Prepare wandb and csv loggers.
    logger.info("Initializing wandb logger.")
    wandb_logger = WandbLogger(
        name=config.loggers.wandb.name,
        save_dir=config.loggers.wandb.save_dir,
        offline=config.loggers.wandb.offline,
        id=config.loggers.wandb.id,
        anonymous=config.loggers.wandb.anonymous,
        version=config.loggers.wandb.version,
        project=config.loggers.wandb.project,
        log_model=config.loggers.wandb.log_model,
        experiment=config.loggers.wandb.experiment,
        prefix=config.loggers.wandb.prefix,
        **resolve_omegaconf_to_dictionary(config.loggers.wandb.init_kwargs),
    )
    logger.info("Initializing csv logger.")
    csv_logger = CSVLogger(
        config.loggers.csv.save_dir,
        name=config.loggers.csv.name,
        version=config.loggers.csv.version,
        prefix=config.loggers.csv.prefix,
        flush_logs_every_n_steps=config.loggers.csv.flush_logs_every_n_steps,
    )
    training_loggers = [wandb_logger, csv_logger]

    # Prepare callbacks.
    logger.info("Initializing early stopping callback.")
    early_stopping = EarlyStopping(
        monitor=config.callbacks.early_stopping.monitor,
        min_delta=config.callbacks.early_stopping.min_delta,
        patience=config.callbacks.early_stopping.patience,
        verbose=config.callbacks.early_stopping.verbose,
        mode=config.callbacks.early_stopping.mode,
        strict=config.callbacks.early_stopping.strict,
        check_finite=config.callbacks.early_stopping.check_finite,
        stopping_threshold=config.callbacks.early_stopping.stopping_threshold,
        divergence_threshold=config.callbacks.early_stopping.divergence_threshold,
        check_on_train_epoch_end=config.callbacks.early_stopping.check_on_train_epoch_end,
    )
    logger.info("Initializing gradient accumulation callback.")
    grad_accumulation = GradientAccumulationScheduler(scheduling=config.callbacks.grad_accumulation.scheduling)
    logger.info("Initializing learning rate monitor callback.")
    lr_monitor = LearningRateMonitor(
        logging_interval=config.callbacks.lr_monitor.logging_interval,
        log_momentum=config.callbacks.lr_monitor.log_momentum,
    )
    logger.info("Initializing model checkpoint callback.")
    model_checkpoint = ModelCheckpoint(
        dirpath=config.callbacks.model_checkpoint.dirpath,
        filename=config.callbacks.model_checkpoint.filename,
        monitor=config.callbacks.model_checkpoint.monitor,
        verbose=config.callbacks.model_checkpoint.verbose,
        save_last=config.callbacks.model_checkpoint.save_last,
        save_top_k=config.callbacks.model_checkpoint.save_top_k,
        save_weights_only=config.callbacks.model_checkpoint.save_weights_only,
        mode=config.callbacks.model_checkpoint.mode,
        auto_insert_metric_name=config.callbacks.model_checkpoint.auto_insert_metric_name,
        every_n_train_steps=config.callbacks.model_checkpoint.every_n_train_steps,
        train_time_interval=config.callbacks.model_checkpoint.train_time_interval,
        every_n_epochs=config.callbacks.model_checkpoint.every_n_epochs,
        save_on_train_epoch_end=config.callbacks.model_checkpoint.save_on_train_epoch_end,
    )
    callbacks = [early_stopping, grad_accumulation, lr_monitor, model_checkpoint]

    # Configuring the training plugins.
    plugins = None  # TODO

    # Initialize ``lightning.pytorch.Trainer`` object.
    logger.info("Initializing trainer.")
    trainer = Trainer(
        accelerator=config.trainer.accelerator,
        strategy=config.trainer.strategy,
        devices=config.trainer.devices,
        num_nodes=config.trainer.num_nodes,
        precision=config.trainer.precision,
        logger=training_loggers,
        callbacks=callbacks,
        fast_dev_run=config.trainer.fast_dev_run,
        max_epochs=config.trainer.max_epochs,
        min_epochs=config.trainer.min_epochs,
        max_steps=config.trainer.max_steps,
        min_steps=config.trainer.min_steps,
        max_time=config.trainer.max_time,
        limit_train_batches=config.trainer.limit_train_batches,
        limit_val_batches=config.trainer.limit_val_batches,
        limit_test_batches=config.trainer.limit_test_batches,
        limit_predict_batches=config.trainer.limit_predict_batches,
        overfit_batches=config.trainer.overfit_batches,
        val_check_interval=config.trainer.val_check_interval,
        check_val_every_n_epoch=config.trainer.check_val_every_n_epoch,
        num_sanity_val_steps=config.trainer.num_sanity_val_steps,
        log_every_n_steps=config.trainer.log_every_n_steps,
        enable_checkpointing=config.trainer.enable_checkpointing,
        enable_progress_bar=config.trainer.enable_progress_bar,
        enable_model_summary=config.trainer.enable_model_summary,
        gradient_clip_val=config.trainer.gradient_clip_val,
        gradient_clip_algorithm=config.trainer.gradient_clip_algorithm,
        deterministic=config.trainer.deterministic,
        benchmark=config.trainer.benchmark,
        inference_mode=config.trainer.inference_mode,
        use_distributed_sampler=config.trainer.use_distributed_sampler,
        profiler=config.trainer.profiler,  # TODO: this can be an object, resolve it somehow.
        detect_anomaly=config.trainer.detect_anomaly,
        barebones=config.trainer.barebones,
        plugins=plugins,
        sync_batchnorm=config.trainer.sync_batchnorm,
        reload_dataloaders_every_n_epochs=config.trainer.reload_dataloaders_every_n_epochs,
        default_root_dir=config.trainer.default_root_dir,
        enable_autolog_hparams=config.trainer.enable_autolog_hparams,
        model_registry=config.trainer.model_registry,
    )

    logger.info("Fitting the model.")
    trainer.fit(plan, datamodule=datamodule, ckpt_path=config.trainer.ckpt_path)

    # Save the trained model: its spec and the module's weights.
    model_cfg = ModelConfig(data=dm_cfg, module=module_cfg, training=training_cfg, inference=inference_cfg)
    model = Model(model_cfg, datamodule.dm, datamodule.data_dims, module, training_method, inference_method)
    logger.info(f"Saving model {config.save.run_dir}.")
    model.save_pretrained(config.save.run_dir, overwrite=config.run.allow_overwrite)

    if new_split:
        for p, a in zip(paths, adatas, strict=True):
            logger.info(f"Writing the split to obs[{splitter.split_key!r}] of {p}.")
            write_obs_column(p, splitter.split_key, a.obs[splitter.split_key].array)
    if preprocessed:
        # Into the input file: every cell's projection, and the basis that maps predictions back to genes.
        for p, a in zip(paths, adatas, strict=True):
            logger.info(f"Writing the preprocessing into {p}.")
            write_preprocessing(p, a, config.datamanager.target_sample_rep, overwrite=pp.overwrite)

    # Return the metric the checkpoint watches (e.g. "val/edist"), so a sweep optimizes what the callbacks select on:
    # its best value, or the last logged one when no checkpoint was scored (e.g. under fast_dev_run). NaN when it was
    # never logged, so a sweep marks the trial failed instead of ranking it.
    monitored = config.callbacks.model_checkpoint.monitor
    score = model_checkpoint.best_model_score
    if score is None:
        score = trainer.callback_metrics.get(monitored)
    if score is None:
        logger.warning(
            f"The monitored metric {monitored!r} was never logged (logged: {sorted(trainer.callback_metrics)}); "
            "returning nan."
        )
        return float("nan")
    returned_metric = float(score)
    logger.info(f"Finishing the run with {monitored} of {returned_metric:.4f}")
    return returned_metric


if __name__ == "__main__":
    main()
