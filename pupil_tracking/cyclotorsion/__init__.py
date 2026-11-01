"""Cyclotorsion measurement package (additive feature).

Modules
-------
pentacam_loader     : Pentacam image import, auto-flip and iris-ROI helpers.
polar_unwrap        : Fast vectorised iris polar unwrapping (Daugman rubber-sheet).
iris_features       : Rotation-aware features/profile on the unwrapped strip.
cyclotorsion_engine : Signed torsional angle estimation from strip correlation.
pair_readout        : Display-only OD/OS pair readout (loader + unwrap + engine).

Sub-modules are imported explicitly by callers.  This package intentionally
performs no eager imports so that lightweight users (e.g. the Add PX dialog)
do not pull in heavy optional dependencies (torch, scikit-image) just by
importing a single helper.
"""
