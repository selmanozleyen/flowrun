"""Persist a fitted transform alongside a saved run.

Two files, mirroring the config/runtime split of the transforms themselves:

* ``transform.json`` — the portable ``TransformConfig`` spec
  (``{"type": ..., ...fields, "version": ...}``). Small, JSON-safe, records
  *which* transform was applied and its hyperparameters.
* ``transform_params.npz`` — the fitted numpy arrays (``mean``, ``std``,
  ``components``, ``W``, ``iW``, ...). ``np.savez_compressed`` is numpy's own
  pickle-free format, so nothing arbitrary is deserialized on load.

These helpers are deliberately separate from :meth:`sckitflow.Model.save_pretrained` /
:class:`sckitflow.ModelConfig`: whether ``ModelConfig`` should grow a ``transform``
field is a design decision still under discussion. When that decision lands,
:func:`save_transform` can be deleted and its two writes folded into
``save_pretrained``; until then this module is the persistence boundary.

Usage from a training script::

    from flowrun.preprocessing.transforms import save_transform, load_transform

    save_transform(config.save.run_dir, transform_cfg, transform)
    # ... later, in a different process:
    transform = load_transform(config.save.run_dir)
"""

import json
from dataclasses import fields
from pathlib import Path

import numpy as np

from flowrun.preprocessing.transforms._base import (
    BaseTransform,
    TransformConfig,
    TransformParams,
)

__all__ = [
    "TRANSFORM_PARAMS_FILENAME",
    "TRANSFORM_SPEC_FILENAME",
    "load_transform",
    "save_transform",
]

TRANSFORM_SPEC_FILENAME = "transform.json"
TRANSFORM_PARAMS_FILENAME = "transform_params.npz"


def save_transform(
    run_dir: str | Path,
    transform_cfg: TransformConfig,
    transform: BaseTransform,
) -> None:
    """Persist a fitted transform to ``run_dir`` as ``transform.json`` + ``transform_params.npz``.

    Args:
        run_dir: Directory to write into. Created (including parents) if it
            does not exist.
        transform_cfg: The portable config that describes the transform.
            Serialized via ``to_spec()`` — the same dict ``Model.save_pretrained``
            embeds inside ``config.json``.
        transform: The fitted runtime transform. Must have been ``fit`` /
            ``fit_adata``; ``params`` must be a single :class:`TransformParams`
            instance (not a collection — no transform currently uses one, but
            the check documents the assumption).

    Raises
    ------
        RuntimeError: If ``transform.is_fitted`` is ``False``.
        TypeError: If ``transform.params`` is not a single
            :class:`TransformParams` instance.
    """
    if not transform.is_fitted:
        raise RuntimeError(
            f"Cannot save an unfitted transform ({type(transform).__name__}); call ``fit`` or ``fit_adata`` first."
        )

    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    # 1. Spec — portable metadata.
    spec_path = run_dir / TRANSFORM_SPEC_FILENAME
    spec_path.write_text(json.dumps(transform_cfg.to_spec(), indent=2))

    # 2. Params — fitted arrays only, no pickle.
    params = transform.params
    if not isinstance(params, TransformParams):
        raise TypeError(
            f"{type(transform).__name__}.params is a collection of "
            f"TransformParams ({type(params).__name__}); "
            "``save_transform`` only supports transforms whose params "
            "is a single TransformParams instance."
        )
    arrays = {
        f.name: np.asarray(getattr(params, f.name)) for f in fields(params) if getattr(params, f.name) is not None
    }
    params_path = run_dir / TRANSFORM_PARAMS_FILENAME
    np.savez_compressed(params_path, **arrays)


def load_transform(run_dir: str | Path) -> BaseTransform:
    """Reconstruct a fitted transform from the files written by :func:`save_transform`.

    Reads ``transform.json`` to rebuild the config via
    ``TransformConfig.from_spec`` (family-base dispatch on the ``type`` field),
    calls ``build()`` for a fresh runtime transform, then restores ``_params``
    from ``transform_params.npz``. ``np.savez_compressed`` round-trips
    ``float32`` / ``float64`` / ``int`` / ``bool`` arrays exactly, so the
    result is numerically identical to the transform the training run used.

    Args:
        run_dir: Directory that :func:`save_transform` wrote into.

    Returns
    -------
        A fitted :class:`BaseTransform` ready for ``transform`` /
        ``inverse_transform``.

    Raises
    ------
        FileNotFoundError: If either file is missing under ``run_dir``.
        ValueError: If the loaded transform's ``_params_cls`` is unset — an
            indication the runtime transform is a custom subclass that did
            not declare its params dataclass.
    """
    run_dir = Path(run_dir)
    spec_path = run_dir / TRANSFORM_SPEC_FILENAME
    params_path = run_dir / TRANSFORM_PARAMS_FILENAME

    if not spec_path.is_file():
        raise FileNotFoundError(f"Missing transform spec at {spec_path}.")
    if not params_path.is_file():
        raise FileNotFoundError(f"Missing transform params at {params_path}.")

    spec = json.loads(spec_path.read_text())
    # ``TransformConfig.from_spec`` dispatches on the ``type`` field of
    # ``spec`` (scfit's ``Component`` family-base contract) and returns the
    # concrete ``*TransformConfig``.
    transform_cfg = TransformConfig.from_spec(spec)
    transform = transform_cfg.build()

    params_cls = type(transform)._params_cls
    if params_cls is None:
        raise ValueError(
            f"{type(transform).__name__} does not declare ``_params_cls``; "
            "``load_transform`` cannot reconstruct its fitted state."
        )

    with np.load(params_path) as loaded:
        # ``np.savez`` stores everything as arrays, including scalar fields
        # like ``is_fitted`` / ``n_features``. ``.item()`` on 0-d arrays
        # recovers the original Python scalar so dataclass fields stay typed.
        kwargs = {key: (loaded[key].item() if loaded[key].ndim == 0 else loaded[key]) for key in loaded.files}
    # ``_params`` is protected; assignment here is deliberate — this module
    # is the persistence boundary and the long-term home for a public
    # ``BaseTransform.from_params`` classmethod.
    transform._params = params_cls(**kwargs)

    return transform
