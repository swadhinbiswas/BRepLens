"""Mock NXOpen.Features submodule (re-exports the fake feature namespace)."""

from . import Features as _features

Feature = _features.Feature
HolePackage = _features.HolePackage
HolePackageBuilder = _features.HolePackageBuilder
BlockFeatureBuilder = _features.BlockFeatureBuilder
CylinderBuilder = _features.CylinderBuilder
