"""Exceptions and CAD adapter contract.

Core modules depend only on this interface and on :mod:`breplens.schema`.
Nothing outside ``breplens.adapters`` may import OCP, CadQuery or NXOpen.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from breplens.schema import FeatureSet, MeshData, PartGraph


class BRepLensError(Exception):
    """Base class for all application errors."""


class CadLoadError(BRepLensError):
    """The CAD file could not be read or contains no solid body."""


class UnsupportedFormatError(BRepLensError):
    """The file extension has no registered adapter."""


class RecognitionError(BRepLensError):
    """Feature recognition failed unexpectedly."""


class ValidationError(BRepLensError):
    """Input violates a documented limit or validation rule."""


class WriteBackError(BRepLensError):
    """Parametric write-back failed (NX only)."""


class CadAdapter(ABC):
    """Abstract adapter between a CAD kernel and the neutral BRepLens IR.

    Implementations must be stateless apart from caches: one adapter instance
    may be reused across parts. ``load_graph`` is the only method required by
    the recognition pipeline; ``tessellate`` is used by the viewer and
    ``write_features`` only by CAD systems that support parametric modelling.
    """

    name: str = "abstract"
    supported_suffixes: tuple[str, ...] = (".step", ".stp")

    @abstractmethod
    def load_graph(self, path: str | Path, *, name: str | None = None) -> PartGraph:
        """Read a part and return its Attributed Adjacency Graph."""

    @abstractmethod
    def tessellate(
        self, path: str | Path, *, deflection: float = 0.2
    ) -> MeshData:
        """Tessellate the first solid in *path* for visualization."""

    def write_features(self, path: str | Path, features: FeatureSet) -> str:
        """Write features back as native parametric features.

        Only adapters with a modelling API implement this; the default is a
        clear, typed failure rather than a silent no-op.
        """
        raise WriteBackError(f"{self.name} adapter does not support write-back")
