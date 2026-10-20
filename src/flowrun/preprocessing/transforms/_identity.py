"""Identity (no-op) transform.

A pass-through transform: ``transform`` and ``inverse_transform`` return a
copy of the input. Useful as

* a default/placeholder wherever a ``TransformConfig`` is required but no
  transformation is wanted (e.g. the ``pred_transform`` / ``target_transform``
  slots in ``TrainingPlan``), or
* a "do nothing" node in a chained pipeline so a chain stays homogeneous.

Unlike every other transform, identity does **not** need to be fitted:

* ``_params`` is set to ``IdentityParams(is_fitted=True)`` at construction,
  so ``is_fitted`` is always ``True`` and ``params`` never raises,
* ``transform`` and ``inverse_transform`` only validate the input shape, not
  "has this been fitted?" or "does the width match the fitted width" — for a
  no-op there is nothing to compare against.

``fit`` is retained for API parity (pipelines that call ``fit`` on every
transform keep working) and, if called, records ``n_features`` on ``_params``
so downstream code that inspects ``params.n_features`` sees a populated field
when a fit did happen. Calling ``fit`` is optional, and never required for
``transform`` to work.

``transform`` returns a **copy** rather than the input itself. This keeps the
invariant every other :class:`BaseTransform` follows (the returned array
never aliases the input), which matters for the AnnData adapter: writing an
identity output to ``adata.obsm["X_id"]`` must not alias ``adata.X``.

See :class:`IdentityTransformConfig` for the portable, serializable side.
"""

from dataclasses import dataclass

import numpy as np
from scfit.registry import component

from flowrun.preprocessing.transforms._base import (
    BaseTransform,
    TransformConfig,
    TransformParams,
)

__all__ = ["IdentityParams", "IdentityTransform", "IdentityTransformConfig"]


@dataclass
class IdentityParams(TransformParams):
    """Fitted parameters for the identity transform.

    Identity has no learned parameters — the only fields ever populated are
    the base-class ``is_fitted`` (always ``True``) and ``n_features`` (set by
    ``fit`` when it is called; ``None`` otherwise).
    """


class IdentityTransform(BaseTransform):
    """No-op transform. Always ready to apply — fitting is optional.

    ``transform`` and ``inverse_transform`` return a copy of the input and
    validate only the input shape (2-D, non-empty). They do not check the
    fitted state or the width, because identity has neither a learned state
    nor a meaningful width to compare against.

    ``fit`` records ``n_features`` for callers that inspect
    ``params.n_features`` after a fit; it does not affect ``transform``.
    """

    _params_cls = IdentityParams

    def __init__(self) -> None:
        super().__init__()
        # Identity is fitted from birth — there is no state to learn.
        # Setting ``_params`` here (rather than leaving it ``None``) makes
        # ``is_fitted`` return ``True`` and lets ``params`` hand back a
        # usable :class:`IdentityParams` without a prior ``fit``.
        self._params: IdentityParams = IdentityParams(is_fitted=True)

    def fit(self, X: np.ndarray) -> "IdentityTransform":
        """Record ``n_features`` and return ``self`` — no state is learned.

        Optional: :meth:`transform` works without a prior ``fit``.

        Args:
            X: Input data array, shape ``(n_samples, n_features)``.

        Returns
        -------
            Self
        """
        self._validate_input(X)
        self._params = IdentityParams(is_fitted=True, n_features=X.shape[1])
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Return a copy of ``X``.

        Args:
            X: Input data array, shape ``(n_samples, n_features)``.

        Returns
        -------
            A copy of ``X``.
        """
        self._validate_input(X)
        return X.copy()

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """Return a copy of ``X`` — inverse of identity is identity.

        Args:
            X: Input data array, shape ``(n_samples, n_features)``.

        Returns
        -------
            A copy of ``X``.
        """
        self._validate_input(X)
        return X.copy()

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Convenience method — identical to :meth:`transform` for identity."""
        return self.transform(X)


@component("transform.identity")
class IdentityTransformConfig(TransformConfig):
    """Portable config for :class:`IdentityTransform`.

    Carries no hyperparameters — the spec is just ``{"type": "transform.identity"}``.

    ``build()`` returns a ready-to-use transform, so identity is the only
    config whose ``build()`` and ``build_and_fit(adata)`` are equivalent
    (fitting is a no-op). Both call sites are supported; neither is required.
    """

    def build(self) -> IdentityTransform:
        """Return a fresh, ready-to-use :class:`IdentityTransform`."""
        return IdentityTransform()
