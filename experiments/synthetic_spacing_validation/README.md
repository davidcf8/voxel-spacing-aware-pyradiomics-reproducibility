# Synthetic Spacing Validation

This folder contains the synthetic validation experiment for the voxel-spacing-aware PyRadiomics implementation.

The experiment uses synthetic anisotropic and isotropic NIfTI images and masks. It extracts Original-image texture features for GLCM, GLRLM, GLDM, NGTDM, and GLSZM and compares:

- default behavior against the upstream-compatible path;
- default anisotropic extraction against `weightingNorm=voxelSpacing`;
- isotropic default extraction against isotropic `weightingNorm=voxelSpacing`.

The validation is a software behavior test, not a clinical validation experiment. It verifies activation of spacing-aware paths under controlled anisotropic spacing and compatibility of standard behavior when the extension is disabled.

The top-level `rerun_experiments.py` script is the preferred entry point for public end-to-end reproduction.
