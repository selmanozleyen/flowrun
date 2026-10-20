import abc
from collections.abc import Collection
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

import numpy as np
from scfit.registry import Component

if TYPE_CHECKING:
    from anndata import AnnData

__all__ = ["BaseTransform", "TransformConfig", "TransformParams"]


@dataclass
class TransformParams:
    """Base class for storing fitted preprocessor parameters.

    This is a container for parameters learned during the fit process.
    Each specific preprocessor will define its own parameter class.

    Attributes
    ----------
        is_fitted: Whether the preprocessor has been fitted to data.
        n_features: Number of features in the fitted data.
    """

    is_fitted: bool = False
    n_features: int | None = None

    def _validate_fitted(self) -> None:
        """Check that the preprocessor has been fitted.

        Raises
        ------
            RuntimeError: If the preprocessor has not been fitted yet.
        """
        if not self.is_fitted:
            raise RuntimeError(
                "This preprocessor has not been fitted yet. "
                "Call 'fit' or 'fit_transform' before using 'transform' or 'inverse_transform'."
            )

    def _validate_n_features(self, X: np.ndarray) -> None:
        """Check that input has correct number of features.

        Args:
            X: Input array to validate, shape (n_samples, n_features).

        Raises
        ------
            ValueError: If number of features doesn't match fitted data.
        """
        if X.shape[1] != self.n_features:
            raise ValueError(
                f"Input has {X.shape[1]} features, but preprocessor was fitted with {self.n_features} features."
            )


class BaseTransform(abc.ABC):
    """Abstract base class for all preprocessors.

    Preprocessors work with :class:`numpy` objects of shape
    ``(n_samples, n_features)``.

    The derived classes will need to override the following abstract methods
    to be instantiated:

        - ``_fit``: Learn parameters from data
        - ``_transform``: Apply transformation using learned parameters
        - ``_inverse_transform``: Reverse the transformation

    An AnnData adapter sits on top of the numpy API: ``fit_adata``,
    ``transform_adata``, ``inverse_transform_adata`` and
    ``fit_transform_adata`` read/write a matrix stored in ``adata.X``,
    ``adata.layers[layer]`` or ``adata.obsm[obsm]``. See :class:`TransformConfig`
    for the portable, serializable side of the same transformations.

    Subclasses that want to be persistable via
    :func:`flowrun.preprocessing.transforms.save_transform` must set
    :attr:`_params_cls` to their concrete :class:`TransformParams` dataclass.
    """

    # Which ``TransformParams`` dataclass this transform populates on ``_params``.
    # Concrete subclasses set this to their specific params class; the base
    # leaves it ``None`` so a custom subclass that forgets to declare it is
    # caught at persistence time rather than silently mis-roundtripped.
    _params_cls: ClassVar[type[TransformParams] | None] = None

    def __init__(self) -> None:
        """Initialize the preprocessor.

        The params attribute will store fitted parameters.
        It should be set to a specific TransformParams subclass in subclasses.
        """
        self._params: TransformParams | Collection[TransformParams] | None = None

    @property
    def params(self) -> TransformParams | Collection[TransformParams]:
        """Get the fitted parameters.

        Returns
        -------
            The fitted parameters object.

        Raises
        ------
            RuntimeError: If the preprocessor has not been fitted yet.
        """
        if self._params is None:
            raise RuntimeError("Preprocessor has not been fitted yet. Call 'fit()' or 'fit_transform()' first.")
        return self._params

    @property
    def is_fitted(self) -> bool:
        """Check if the preprocessor has been fitted.

        Returns
        -------
            True if fitted, False otherwise.
        """
        return self._params is not None and self._params.is_fitted

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit to data, then transform it.

        This is a convenience method that combines fit and transform.

        Args:
            X: Input data array, shape (n_samples, n_features).

        Returns
        -------
            Transformed data array, shape (n_samples, n_features_out).
        """
        self.fit(X)
        return self.transform(X)

    def _validate_input(self, X: np.ndarray, name: str = "X") -> None:
        """Validate input array shape and type.

        Args:
            X: Input array to validate.
            name: Name of the parameter for error messages.

        Raises
        ------
            TypeError: If X is not a numpy array.
            ValueError: If X is not 2D or has invalid shape.
        """
        if not isinstance(X, np.ndarray):
            raise TypeError(f"{name} must be a numpy array, got {type(X)}")

        if X.ndim != 2:
            raise ValueError(f"{name} must be 2D array with shape (n_samples, n_features), got shape {X.shape}")

        if X.shape[0] == 0:
            raise ValueError(f"{name} must have at least one sample")

        if X.shape[1] == 0:
            raise ValueError(f"{name} must have at least one feature")

    # ---- AnnData adapters --------------------------------------------------

    @staticmethod
    def _resolve_view(
        adata: "AnnData",
        *,
        layer: str | None,
        obsm: str | None,
    ) -> np.ndarray:
        """Return the matrix selected by ``layer`` / ``obsm`` (``obsm`` wins)."""
        if layer is not None and obsm is not None:
            raise ValueError("Pass at most one of 'layer' or 'obsm'.")
        if obsm is not None:
            return adata.obsm[obsm]
        if layer is not None:
            return adata.layers[layer]
        return adata.X

    @staticmethod
    def _write_view(
        adata: "AnnData",
        values: np.ndarray,
        *,
        layer: str | None,
        obsm: str | None,
    ) -> None:
        """Write ``values`` back into ``adata`` at the selected slot."""
        if layer is not None and obsm is not None:
            raise ValueError("Pass at most one of 'layer' or 'obsm'.")
        if obsm is not None:
            adata.obsm[obsm] = values
            return
        if layer is not None:
            adata.layers[layer] = values
            return
        adata.X = values

    @staticmethod
    def _resolve_write_slot(
        *,
        layer: str | None,
        obsm: str | None,
        out_layer: str | None,
        out_obsm: str | None,
    ) -> tuple[str | None, str | None]:
        """Resolve where the result of a transform should be written.

        If either ``out_layer`` or ``out_obsm`` is provided, that slot is used
        and the input slot is *not* reused — so ``layer="counts",
        out_obsm="X_pca"`` writes only to ``obsm``. If neither ``out_*`` is
        given, the input slot is overwritten (the default "in-place" contract).

        Raises
        ------
            ValueError: If both ``out_layer`` and ``out_obsm`` are provided.
        """
        if out_layer is not None and out_obsm is not None:
            raise ValueError("Pass at most one of 'out_layer' or 'out_obsm'.")
        if out_layer is not None or out_obsm is not None:
            return out_layer, out_obsm
        return layer, obsm

    def fit_adata(
        self,
        adata: "AnnData",
        *,
        layer: str | None = None,
        obsm: str | None = None,
    ) -> "BaseTransform":
        """Fit the transform on a matrix read from ``adata``.

        Reads ``adata.obsm[obsm]`` if ``obsm`` is set, else
        ``adata.layers[layer]`` if ``layer`` is set, else ``adata.X``.
        Returns ``self`` so calls chain like the numpy ``fit``.
        """
        return self.fit(self._resolve_view(adata, layer=layer, obsm=obsm))

    def transform_adata(
        self,
        adata: "AnnData",
        *,
        layer: str | None = None,
        obsm: str | None = None,
        out_layer: str | None = None,
        out_obsm: str | None = None,
        inplace: bool = True,
    ) -> "AnnData":
        """Transform ``adata`` in place (or on a copy).

        ``layer`` / ``obsm`` select the input matrix (same rules as
        :meth:`fit_adata`). ``out_layer`` / ``out_obsm`` select where the
        result is written; if both are ``None``, the input slot is
        overwritten. ``out_layer`` and ``out_obsm`` are mutually exclusive —
        pass at most one.

        The ``layer → out_obsm`` combination (read from a layer, write to
        ``obsm``) is supported and is the common "reduce to PC space while
        keeping raw values intact" pattern.

        When ``inplace=False`` the transform is applied to a copy, and that
        copy is returned.
        """
        target = adata if inplace else adata.copy()
        X = self.transform(self._resolve_view(target, layer=layer, obsm=obsm))
        write_layer, write_obsm = self._resolve_write_slot(
            layer=layer, obsm=obsm, out_layer=out_layer, out_obsm=out_obsm
        )
        self._write_view(target, X, layer=write_layer, obsm=write_obsm)
        return target

    def inverse_transform_adata(
        self,
        adata: "AnnData",
        *,
        layer: str | None = None,
        obsm: str | None = None,
        out_layer: str | None = None,
        out_obsm: str | None = None,
        inplace: bool = True,
    ) -> "AnnData":
        """Reverse-transformation adapter mirroring :meth:`transform_adata`."""
        target = adata if inplace else adata.copy()
        X = self.inverse_transform(self._resolve_view(target, layer=layer, obsm=obsm))
        write_layer, write_obsm = self._resolve_write_slot(
            layer=layer, obsm=obsm, out_layer=out_layer, out_obsm=out_obsm
        )
        self._write_view(target, X, layer=write_layer, obsm=write_obsm)
        return target

    def fit_transform_adata(
        self,
        adata: "AnnData",
        *,
        layer: str | None = None,
        obsm: str | None = None,
        out_layer: str | None = None,
        out_obsm: str | None = None,
        inplace: bool = True,
    ) -> "AnnData":
        """Fit and transform in one call, mirroring :meth:`fit_transform`."""
        self.fit_adata(adata, layer=layer, obsm=obsm)
        return self.transform_adata(
            adata,
            layer=layer,
            obsm=obsm,
            out_layer=out_layer,
            out_obsm=out_obsm,
            inplace=inplace,
        )

    @abc.abstractmethod
    def fit(self, X: np.ndarray) -> "BaseTransform":
        """Fit the preprocessor to the data.

        This method learns parameters from the input data and stores them
        in self._params. It should:
        1. Validate input shape
        2. Compute statistics/parameters
        3. Store parameters in a _TransformParams object
        4. Set is_fitted=True
        5. Return self for method chaining

        Args:
            X: Input data array, shape (n_samples, n_features).

        Returns
        -------
            Self

        Raises
        ------
            ValueError: If input array has invalid shape.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform the data using fitted parameters.

        This method applies the learned transformation to new data.
        It should:
        1. Check that preprocessor is fitted
        2. Validate input shape matches fitted data
        3. Apply transformation
        4. Return transformed array

        Args:
            X: Input data array, shape (n_samples, n_features).

        Returns
        -------
            Transformed data array, shape (n_samples, n_features_out).

        Raises
        ------
            RuntimeError: If preprocessor has not been fitted.
            ValueError: If input shape is incompatible.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """Reverse the transformation.

        This method reverses the transformation to recover (approximately)
        the original data space. It should:
        1. Check that preprocessor is fitted
        2. Validate input shape
        3. Apply inverse transformation
        4. Return reconstructed array

        Args:
            X: Transformed data array, shape (n_samples, n_features_out).

        Returns
        -------
            Reconstructed data array, shape (n_samples, n_features).

        Raises
        ------
            RuntimeError: If preprocessor has not been fitted.
            ValueError: If input shape is incompatible.
        """
        raise NotImplementedError


class TransformConfig(Component):
    """Family base for portable transform configs.

    Subclasses hold only portability-safe fields — ints, floats, strings,
    ``None`` — and ``build()`` returns a fresh :class:`BaseTransform`. The
    runtime object (with its fitted numpy arrays in ``_params``) is never
    serialized: only the hyperparameters in the config end up in a
    ``config.json`` written by ``Model.save_pretrained``.

    Two usage patterns:

    1. Fully explicit — build, fit, transform::

           cfg = PCATransformConfig(n_components=50)
           xf = cfg.build()
           xf.fit_adata(adata, layer="counts")
           adata = xf.transform_adata(adata, layer="counts", out_obsm="X_pca")

    2. One-shot fit — build + fit in a single call::

           xf = PCATransformConfig(n_components=50).build_and_fit(adata, layer="counts")
    """

    def build(self) -> BaseTransform:
        """Return a fresh, unfitted transform instance."""
        raise NotImplementedError(
            f"{type(self).__name__} does not implement build(); "
            "subclass TransformConfig and return a BaseTransform instance."
        )

    def build_and_fit(
        self,
        adata: "AnnData",
        *,
        layer: str | None = None,
        obsm: str | None = None,
    ) -> BaseTransform:
        """Build a fresh transform and fit it on ``adata`` in one call.

        Convenience over ``self.build().fit_adata(adata, ...)``; the returned
        object is a fitted runtime :class:`BaseTransform`, not a new config.
        """
        return self.build().fit_adata(adata, layer=layer, obsm=obsm)
